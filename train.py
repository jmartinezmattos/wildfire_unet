import os
import time
import json
from datetime import datetime
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Use values
LEARNING_RATE = config["LEARNING_RATE"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = config["BATCH_SIZE"]
NUM_EPOCHS = config["NUM_EPOCHS"]
NUM_WORKERS = config["NUM_WORKERS"]
IMAGE_HEIGHT = config["IMAGE_HEIGHT"]
IMAGE_WIDTH = config["IMAGE_WIDTH"]
PIN_MEMORY = config["PIN_MEMORY"]
LOAD_MODEL = config["LOAD_MODEL"]
TRAIN_IMG_DIR = config["TRAIN_IMG_DIR"]
TRAIN_MASK_DIR = config["TRAIN_MASK_DIR"]
VAL_IMG_DIR = config["VAL_IMG_DIR"]
VAL_MASK_DIR = config["VAL_MASK_DIR"]
MODEL = config["MODEL"] # custom or smp

if MODEL not in ["custom", "smp"]:
    raise ValueError("MODEL must be either 'custom' or 'smp'")

print(f"Using device: {DEVICE}")
print(f"Model type: {MODEL}")

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
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

    if MODEL == "custom":
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    elif MODEL == "smp":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        ).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoints/my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    train_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"logs/log_{train_timestamp}.txt"

    start_time = time.time()

    with open(log_file, "w") as f:
        f.write(f"Training started at: {datetime.now()}\n")
        f.write(f"Hyperparameters: {json.dumps(config)}\n\n")

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # Guardar modelo
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        save_checkpoint(checkpoint, filename=f"my_checkpoint_{train_timestamp}.pth.tar")

        # Check accuracy
        check_accuracy(val_loader, model, device=DEVICE)


    total_time = time.time() - start_time

    with open(log_file, "a") as f:
        f.write(f"\nTraining finished at: {datetime.now()}\n")
        f.write(f"Total training time: {total_time/60:.2f} minutes\n")


if __name__ == "__main__":
    main()