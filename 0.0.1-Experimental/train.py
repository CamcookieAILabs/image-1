import csv
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from model import ConditionalUNet


# -----------------------------
# Configuration
# -----------------------------

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 256
BATCH_SIZE = 4 if DEVICE.type == "cuda" else 2
EPOCHS = 100
TIMESTEPS = 1000
LEARNING_RATE = 0.0001
MAX_LENGTH = 80

DATASET = Path(__file__).resolve().parent / "dataset"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

CHECKPOINT_DIR.mkdir(exist_ok=True)


# -----------------------------
# Build vocabulary
# -----------------------------

with open(DATASET / "captions.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise RuntimeError("No captions found.")

vocab = {
    char: index + 1
    for index, char in enumerate(
        sorted(set(
            "".join(row["caption"].lower() for row in rows)
        ))
    )
}


def tokenize(text):

    ids = [
        vocab.get(char, 0)
        for char in text.lower()
    ][:MAX_LENGTH]

    ids += [0] * (MAX_LENGTH - len(ids))

    return torch.tensor(ids, dtype=torch.long)


# -----------------------------
# Dataset
# -----------------------------

class ImageDataset(Dataset):
    def __init__(self):
        self.rows = rows
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            )
        ])
        for row in self.rows:
            image_path = DATASET / "images" / row["filename"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = DATASET / "images" / row["filename"]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        caption = tokenize(row["caption"])
        return image, caption


dataset = ImageDataset()

if len(dataset) < 2:
    raise RuntimeError("Please provide at least 2 images.")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# -----------------------------
# Initialize model
# -----------------------------

model = ConditionalUNet(
    vocab_size=len(vocab) + 1
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Diffusion scheduler
# -----------------------------

betas = torch.linspace(
    0.0001,
    0.02,
    TIMESTEPS,
    device=DEVICE
)

alphas = 1.0 - betas

alpha_bars = torch.cumprod(
    alphas,
    dim=0
)


# -----------------------------
# Training loop
# -----------------------------

print("Starting training...")
print("Device:", DEVICE)
print("Images:", len(dataset))
print("CPU threads:", torch.get_num_threads())

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for images, captions in progress:

        images = images.to(DEVICE)
        captions = captions.to(DEVICE)
        batch_size = images.shape[0]

        t = torch.randint(
            0,
            TIMESTEPS,
            (batch_size,),
            device=DEVICE
        )

        noise = torch.randn_like(images)
        alpha_bar = alpha_bars[t][:, None, None, None]

        noisy_images = (
            torch.sqrt(alpha_bar) * images
            +
            torch.sqrt(1.0 - alpha_bar) * noise
        )

        predicted_noise = model(
            noisy_images,
            t,
            captions
        )

        loss = F.mse_loss(
            predicted_noise,
            noise
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )

        optimizer.step()

        total_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}: loss = {average_loss:.5f}"
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab,
            "resolution": IMAGE_SIZE,
            "timesteps": TIMESTEPS,
        },
        CHECKPOINT_DIR / "latest.pt"
    )

print("Training complete!")