import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import torchvision
from pathlib import Path
from model import UNET
from utils import load_checkpoint

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/my_checkpoint_20251018_235853.pth.tar"

# Use the same dimensions as in training
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240


def predict(model, image_path, transform, device):
    """
    Runs prediction on a single image.
    """
    # Set model to evaluation mode
    model.eval()

    # 1. Load and preprocess the image
    image = np.array(Image.open(image_path).convert("RGB"))
    augmentations = transform(image=image)
    image = augmentations["image"].to(device)

    # 2. Add batch dimension and predict
    with torch.no_grad():
        # Input tensor shape: (1, 3, H, W)
        preds = torch.sigmoid(model(image.unsqueeze(0)))
        preds = (preds > 0.5).float()

    # 3. Save the output mask
    os.makedirs("data/predictions", exist_ok=True)
    output_filename = f"data/predictions/{image_path.stem}_pred.png"
    torchvision.utils.save_image(preds, output_filename)
    print(f"Prediction saved to {output_filename}")


def main():
    """
    Main function to load model and run prediction.
    """
    # 1. Define transformations (must be same as validation)
    transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # 2. Load the trained model
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        load_checkpoint(checkpoint, model)
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at '{MODEL_PATH}'")
        return

    # 3. Run prediction on an image
    # IMPORTANT: Replace this with the actual path to your image!
    
    image_to_predict = Path(r"data\SatFire\SatFire Dataset\test\data\1217_fire.png") # Example image
    predict(model, image_to_predict, transform, DEVICE)

if __name__ == "__main__":
    main()
