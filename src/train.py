
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import (
    CamVidDataset,
    CLASS_NAMES,
    IGNORE_INDEX,
)
from src.model import build_model


# ============================================================
# 1. Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_PATH = MODELS_DIR / "best_segformer_camvid.pth"

DEVICE = torch.device("cpu")

BATCH_SIZE = 2
NUM_EPOCHS = 5
LEARNING_RATE = 0.00005
NUM_WORKERS = 0

IMAGE_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1)

IMAGE_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1)


# ============================================================
# 2. Preprocessing
# ============================================================

def normalize_images(pixel_values):
    """Normalize images using ImageNet channel statistics."""
    mean = IMAGE_MEAN.to(
        device=pixel_values.device,
        dtype=pixel_values.dtype,
    )
    std = IMAGE_STD.to(
        device=pixel_values.device,
        dtype=pixel_values.dtype,
    )

    return (pixel_values - mean) / std


# ============================================================
# 3. Segmentation loss
# ============================================================

def calculate_loss(logits, labels):
    """
    Resize model logits to mask dimensions and calculate
    cross-entropy loss while ignoring label 255.
    """
    logits = F.interpolate(
        logits,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    return F.cross_entropy(
        logits,
        labels,
        ignore_index=IGNORE_INDEX,
    )


# ============================================================
# 4. Mean Intersection over Union
# ============================================================

@torch.no_grad()
def calculate_miou(confusion_matrix):
    """Calculate mean IoU over classes present in ground truth."""
    true_positives = confusion_matrix.diag()

    ground_truth_count = confusion_matrix.sum(dim=1)
    predicted_count = confusion_matrix.sum(dim=0)

    union = ground_truth_count + predicted_count - true_positives

    valid_classes = ground_truth_count > 0

    if not valid_classes.any():
        return 0.0

    iou = true_positives[valid_classes] / union[
        valid_classes
    ].clamp_min(1)

    return iou.mean().item()


@torch.no_grad()
def update_confusion_matrix(confusion_matrix, logits, labels):
    """Update the confusion matrix, excluding ignored pixels."""
    logits = F.interpolate(
        logits,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    predictions = logits.argmax(dim=1)

    valid = labels != IGNORE_INDEX

    labels = labels[valid]
    predictions = predictions[valid]

    num_classes = confusion_matrix.shape[0]

    encoded = labels * num_classes + predictions

    confusion_matrix += torch.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


# ============================================================
# 5. Validation
# ============================================================

def validate(model, val_loader):
    """Evaluate validation loss and mean IoU."""
    model.eval()

    total_loss = 0.0
    total_samples = 0

    num_classes = model.config.num_labels

    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
        device=DEVICE,
    )

    for batch in tqdm(
        val_loader,
        desc="Validation",
        leave=False,
    ):
        pixel_values = normalize_images(
            batch["pixel_values"].to(DEVICE)
        )

        labels = batch["labels"].to(DEVICE)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

            loss = calculate_loss(
                outputs.logits,
                labels,
            )

        batch_size = labels.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        update_confusion_matrix(
            confusion_matrix,
            outputs.logits,
            labels,
        )

    average_loss = total_loss / max(total_samples, 1)
    miou = calculate_miou(confusion_matrix)

    return average_loss, miou


# ============================================================
# 6. Training
# ============================================================

def train():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CamVid Semantic Segmentation Training")
    print("=" * 60)
    print("Device:", DEVICE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LEARNING_RATE)

    train_dataset = CamVidDataset(split="train")
    val_dataset = CamVidDataset(split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    model = build_model().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    best_miou = -1.0

    for epoch in range(NUM_EPOCHS):
        model.train()

        running_loss = 0.0
        processed_samples = 0

        start_time = time.time()

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
        )

        for batch in progress:
            pixel_values = normalize_images(
                batch["pixel_values"].to(DEVICE)
            )

            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(pixel_values=pixel_values)

            loss = calculate_loss(
                outputs.logits,
                labels,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Training loss is NaN or infinite. "
                    "Stop and inspect the data and model."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = labels.shape[0]

            running_loss += loss.item() * batch_size
            processed_samples += batch_size

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = running_loss / max(processed_samples, 1)

        val_loss, val_miou = validate(
            model,
            val_loader,
        )

        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation mIoU: {val_miou:.4f}")
        print(f"Time: {elapsed / 60:.2f} minutes")

        if val_miou > best_miou:
            best_miou = val_miou

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_miou": best_miou,
                    "class_names": [
                        name
                        for name in CLASS_NAMES
                        if name.lower() != "unlabelled"
                    ],
                    "num_classes": model.config.num_labels,
                    "ignore_index": IGNORE_INDEX,
                },
                CHECKPOINT_PATH,
            )

            print("Saved best model:", CHECKPOINT_PATH)

    print("\nTraining completed.")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print("Checkpoint:", CHECKPOINT_PATH)


if __name__ == "__main__":
    train()
