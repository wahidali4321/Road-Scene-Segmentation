
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import (
    CamVidDataset,
    CLASS_NAMES,
    IGNORE_INDEX,
)
from src.model import build_model


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_segformer_camvid.pth"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Evaluation configuration
DEVICE = torch.device("cpu")
BATCH_SIZE = 1
NUM_WORKERS = 0

NUM_CLASSES = len([
    name for name in CLASS_NAMES
    if name.lower() != "unlabelled"
])

CLASS_NAMES_TRAINABLE = [
    name for name in CLASS_NAMES
    if name.lower() != "unlabelled"
]

IMAGE_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1)

IMAGE_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1)


def normalize_images(images):
    mean = IMAGE_MEAN.to(images.device)
    std = IMAGE_STD.to(images.device)
    return (images - mean) / std


def evaluate():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CamVid test dataset...")
    test_dataset = CamVidDataset(split="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("Loading trained SegFormer-B0...")
    model = build_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print(
        "Best validation mIoU:",
        f'{checkpoint.get("best_miou", 0):.4f}',
    )
    print("Test images:", len(test_dataset))

    confusion_matrix = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=torch.int64,
    )

    total_loss = 0.0
    total_batches = 0

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Evaluating test set"):
            images = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            images = normalize_images(images)

            outputs = model(pixel_values=images)
            logits = F.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = F.cross_entropy(
                logits,
                labels,
                ignore_index=IGNORE_INDEX,
            )

            total_loss += loss.item()
            total_batches += 1

            predictions = logits.argmax(dim=1)

            valid = labels != IGNORE_INDEX
            valid_labels = labels[valid]
            valid_predictions = predictions[valid]

            encoded = (
                valid_labels * NUM_CLASSES + valid_predictions
            )

            counts = torch.bincount(
                encoded,
                minlength=NUM_CLASSES * NUM_CLASSES,
            )

            confusion_matrix += counts.reshape(
                NUM_CLASSES, NUM_CLASSES
            )

    diagonal = confusion_matrix.diag().float()
    ground_truth_count = confusion_matrix.sum(dim=1).float()
    prediction_count = confusion_matrix.sum(dim=0).float()

    union = ground_truth_count + prediction_count - diagonal

    class_ious = torch.full(
        (NUM_CLASSES,),
        float("nan"),
    )

    present_classes = ground_truth_count > 0
    class_ious[present_classes] = (
        diagonal[present_classes] / union[present_classes].clamp(min=1)
    )

    miou = class_ious[present_classes].mean().item()

    total_correct = diagonal.sum().item()
    total_valid_pixels = confusion_matrix.sum().item()

    pixel_accuracy = (
        total_correct / total_valid_pixels
        if total_valid_pixels > 0
        else 0.0
    )

    test_loss = total_loss / max(total_batches, 1)

    print("\n" + "=" * 45)
    print("CAMVID TEST RESULTS")
    print("=" * 45)
    print(f"Test loss:       {test_loss:.4f}")
    print(f"Test mIoU:       {miou:.4f} ({miou * 100:.2f}%)")
    print(
        f"Pixel accuracy: {pixel_accuracy:.4f} "
        f"({pixel_accuracy * 100:.2f}%)"
    )

    print("\nPer-class IoU:")
    for index, name in enumerate(CLASS_NAMES_TRAINABLE):
        value = class_ious[index].item()
        if value == value:  # Exclude NaN values
            print(f"{name:15s}: {value:.4f} ({value * 100:.2f}%)")
        else:
            print(f"{name:15s}: Not present in test ground truth")

    # Save per-class results
    results = pd.DataFrame({
        "class": CLASS_NAMES_TRAINABLE,
        "iou": class_ious.tolist(),
        "ground_truth_pixels": ground_truth_count.tolist(),
        "predicted_pixels": prediction_count.tolist(),
    })

    results.to_csv(
        OUTPUTS_DIR / "test_class_iou.csv",
        index=False,
    )

    # Save confusion matrix
    confusion_df = pd.DataFrame(
        confusion_matrix.numpy(),
        index=CLASS_NAMES_TRAINABLE,
        columns=CLASS_NAMES_TRAINABLE,
    )

    confusion_df.to_csv(
        OUTPUTS_DIR / "test_confusion_matrix.csv"
    )

    print("\nSaved results:")
    print(OUTPUTS_DIR / "test_class_iou.csv")
    print(OUTPUTS_DIR / "test_confusion_matrix.csv")
    print("\nTest evaluation completed successfully.")


if __name__ == "__main__":
    evaluate()
