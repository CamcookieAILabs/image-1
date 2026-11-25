import json
import time
import argparse
from pathlib import Path
import torch
from PIL import Image
from model import ConditionalUNet


# ==========================================
# CONFIGURATION
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CPU_THREADS = 4
MAX_LENGTH = 80


CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "latest.pt"
OUTPUT_DIR = Path(__file__).resolve().parent / "generated"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(CPU_THREADS)

if DEVICE.type == "cpu":
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True


# ==========================================
# LOAD CHECKPOINT
# ==========================================

def load_checkpoint():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    required = [
        "vocab",
        "resolution",
        "timesteps",
        "model_state"
    ]

    for key in required:
        if key not in checkpoint:
            raise ValueError(
                f"Checkpoint is missing: {key}"
            )

    return checkpoint


CHECKPOINT = load_checkpoint()

VOCAB = CHECKPOINT["vocab"]
RESOLUTION = int(CHECKPOINT["resolution"])
TIMESTEPS = int(CHECKPOINT["timesteps"])

if not isinstance(VOCAB, dict):
    raise ValueError("Invalid vocabulary in checkpoint")

if RESOLUTION <= 0 or TIMESTEPS <= 0:
    raise ValueError("Invalid checkpoint dimensions")


# ==========================================
# TOKENIZATION
# ==========================================

def tokenize(text):

    if not isinstance(text, str):
        raise TypeError("Prompt must be a string")

    text = text.lower().strip()

    if not text:
        raise ValueError("Prompt cannot be empty")

    ids = [
        VOCAB.get(char, 0)
        for char in text
    ][:MAX_LENGTH]

    ids += [0] * (MAX_LENGTH - len(ids))

    return torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE
    )


# ==========================================
# LOAD MODEL
# ==========================================

def create_model():

    model = ConditionalUNet(
        vocab_size=len(VOCAB) + 1
    ).to(DEVICE)

    model.load_state_dict(
        CHECKPOINT["model_state"],
        strict=True
    )

    model.eval()

    return model


MODEL = create_model()


# ==========================================
# DIFFUSION SCHEDULE
# ==========================================

def create_schedule():

    betas = torch.linspace(
        0.0001,
        0.02,
        TIMESTEPS,
        device=DEVICE,
        dtype=torch.float32
    )

    alphas = 1.0 - betas

    alpha_bars = torch.cumprod(
        alphas,
        dim=0
    )

    previous_alpha_bars = torch.cat([
        torch.ones(
            1,
            device=DEVICE
        ),
        alpha_bars[:-1]
    ])

    posterior_variance = (
        betas
        * (1.0 - previous_alpha_bars)
        / (1.0 - alpha_bars)
    )

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "previous_alpha_bars": previous_alpha_bars,
        "posterior_variance": posterior_variance
    }


SCHEDULE = create_schedule()


# ==========================================
# IMAGE CONVERSION
# ==========================================

def tensor_to_image(tensor):

    tensor = tensor.detach().float()

    tensor = tensor.clamp(-1, 1)

    tensor = (
        (tensor + 1.0) * 127.5
    ).round().byte()

    tensor = tensor[0].permute(
        1, 2, 0
    ).cpu().numpy()

    return Image.fromarray(tensor, mode="RGB")


# ==========================================
# IMAGE GENERATION
# ==========================================

@torch.inference_mode()
def generate(
    prompt,
    seed=None,
    progress=True
):

    tokens = tokenize(prompt)

    generator = torch.Generator(
        device=DEVICE
    )

    if seed is not None:
        generator.manual_seed(seed)

    image = torch.randn(
        (1, 3, RESOLUTION, RESOLUTION),
        generator=generator,
        device=DEVICE
    )

    start_time = time.time()

    betas = SCHEDULE["betas"]
    alphas = SCHEDULE["alphas"]
    alpha_bars = SCHEDULE["alpha_bars"]
    previous = SCHEDULE["previous_alpha_bars"]
    variances = SCHEDULE["posterior_variance"]

    for i in reversed(range(TIMESTEPS)):

        t = torch.tensor(
            [i],
            dtype=torch.long,
            device=DEVICE
        )

        predicted_noise = MODEL(
            image,
            t,
            tokens
        )

        alpha = alphas[i]
        beta = betas[i]
        alpha_bar = alpha_bars[i]
        previous_alpha_bar = previous[i]

        predicted_original = (
            image
            - torch.sqrt(1.0 - alpha_bar) * predicted_noise
        ) / torch.sqrt(alpha_bar)

        predicted_original = predicted_original.clamp(
            -1.0, 1.0
        )

        coef1 = (
            beta
            * torch.sqrt(previous_alpha_bar)
            / (1.0 - alpha_bar)
        )

        coef2 = (
            (1.0 - previous_alpha_bar)
            * torch.sqrt(alpha)
            / (1.0 - alpha_bar)
        )

        image = (
            coef1 * predicted_original
            + coef2 * image
        )

        if i > 0:

            noise = torch.randn(
                image.shape,
                generator=generator,
                device=DEVICE
            )

            image = (
                image
                + torch.sqrt(variances[i]) * noise
            )

        if progress and (
            i == TIMESTEPS - 1
            or i % max(1, TIMESTEPS // 50) == 0
        ):
            completed = TIMESTEPS - i

            print(
                f"Sampling: {completed}/{TIMESTEPS}"
            )

    elapsed = time.time() - start_time

    print(
        f"Generation completed in {elapsed:.2f} seconds"
    )

    return tensor_to_image(image)


# ==========================================
# SAVE IMAGE
# ==========================================

def save_image(image, prompt, seed=None):

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    filename = f"generated_{timestamp}.png"

    output_path = OUTPUT_DIR / filename

    image.save(
        output_path,
        format="PNG",
        optimize=True
    )

    metadata = {
        "prompt": prompt,
        "seed": seed,
        "resolution": RESOLUTION,
        "timesteps": TIMESTEPS,
        "device": str(DEVICE),
        "filename": filename
    }

    metadata_path = output_path.with_suffix(".json")

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(f"\nImage saved: {output_path.resolve()}")

    return output_path


# ==========================================
# MAIN
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Conditional diffusion image generator"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1
    )

    args = parser.parse_args()

    if args.count < 1 or args.count > 100:
        parser.error("--count must be between 1 and 100")

    print("=" * 45)
    print("CAMCOOKIE AI LABS")
    print("IMAGE GENERATOR")
    print("=" * 45)

    print("Device:", DEVICE)
    print("Resolution:", RESOLUTION)
    print("Timesteps:", TIMESTEPS)

    prompt = args.prompt

    if prompt is None:
        prompt = input("\nDescribe your image: ")

    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    for index in range(args.count):

        seed = (
            args.seed + index
            if args.seed is not None
            else None
        )

        print(f"\nGenerating image {index + 1}/{args.count}")

        result = generate(
            prompt,
            seed=seed
        )

        save_image(
            result,
            prompt,
            seed
        )

    print("\nAll generations finished!")


if __name__ == "__main__":
    main()
