import csv
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
import torch
from PIL import Image
import pandas as pd
from pathlib import Path

# Setup
model_name = "nlpconnect/vit-gpt2-image-captioning"

BASE_DIR = Path(__file__).resolve().parent
# load model
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = AutoImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

image_number = "0001"
while int(image_number) <= 257:
    ### replace the value with your image
    img_name = BASE_DIR / "dataset" / "images" / f"car_{image_number}.jpg"

    img = Image.open(img_name)
    if img.mode != 'RGB':
        img = img.convert(mode="RGB")

    pixel_values = feature_extractor(images=[img], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    max_length = 128
    num_beams = 4

    # get model prediction
    output_ids = model.generate(pixel_values, num_beams=num_beams, max_length=max_length)

    # decode the generated prediction
    preds = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Fix preds
    preds = preds.strip()
    if preds:
        preds = preds[0].upper() + preds[1:]
        preds = preds.rstrip(" .") + "."


    # Load CSV
    dataset_path = BASE_DIR / "dataset" / "captions.csv"
    dataset = pd.read_csv(dataset_path)
    # Update CSV
    dataset.loc[dataset["filename"] == f"car_{image_number}.jpg", "caption"] = preds
    # Save updated CSV
    dataset.to_csv(dataset_path, index=False)

    attributions_dataset_path = BASE_DIR / "dataset" / "attributions.csv"
    dataset = pd.read_csv(attributions_dataset_path)
    # Update CSV
    dataset.loc[dataset["filename"] == f"car_{image_number}.jpg", "caption"] = preds
    # Save updated CSV
    dataset.to_csv(attributions_dataset_path, index=False)

    print(f"Image Number: {image_number} Result: {preds}")
    image_number = f"{int(image_number) + 1:04d}"
