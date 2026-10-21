
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.dataset import (
    CamVidDataset,
    CLASS_NAMES,
    COLOR_TO_ID,
    IGNORE_INDEX,
)
from src.model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_segformer_camvid.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"

DEVICE = torch.device("cpu")
IMAGE_SIZE = (360, 480)  # height, width

IMAGE_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1)

IMAGE_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1)

TRAINABLE_CLASSES = [
    name for name in CLASS_NAMES
    if name.lower() != "unlabelled"
]


def create_palette():
    """Create a class-ID to RGB-color mapping from the dataset palette."""
    palette = {}

    for class_id, (color, _) in enumerate(COLOR_TO_ID.items()):
        # COLOR_TO_ID maps RGB colors to class IDs.
        palette[COLOR_TO_ID[color]] = color

    return palette


def colorize_mask(mask):
    """Convert a class-ID mask into an RGB visualization."""
    mask = np.asarray(mask)
    colored = np.zeros(
        (mask.shape[0], mask.shape[1], 3),
        dtype=np.uint8,
    )

    palette = create_palette()

    for class_id, color in palette.items():
        colored[mask == class_id] = color

    # Ignored or unknown pixels appear black.
    colored[mask == IGNORE_INDEX] = (0, 0, 0)

    return Image.fromarray(colored)


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    test_images = sorted((DATA_DIR / "test").glob("*"))

    if not test_images:
        raise FileNotFoundError(
            f"No test images found in {DATA_DIR / 'test'}"
        )

    # Select the first test image for initial verification.
    image_path = test_images[0]
    original = Image.open(image_path).convert("RGB")
    resized = original.resize(
        (IMAGE_SIZE[1], IMAGE_SIZE[0]),
        Image.Resampling.BILINEAR,
    )

    image_tensor = to_tensor(resized).unsqueeze(0)
    image_tensor = (
        image_tensor - IMAGE_MEAN
    ) / IMAGE_STD

    print("Image:", image_path.name)

    with torch.inference_mode():
        outputs = model(pixel_values=image_tensor.to(DEVICE))
        logits = F.interpolate(
            outputs.logits,
            size=IMAGE_SIZE,
            mode="bilinear",
            align_corners=False,
        )
        prediction = logits.argmax(dim=1)[0].cpu().numpy()

    prediction_image = colorize_mask(prediction)

    # Load the matching ground-truth annotation using the dataset class.
    dataset = CamVidDataset(split="test")
    matching_indices = [
        index
        for index in range(len(dataset))
        if Path(dataset.samples[index][0]).name == image_path.name
    ]

    if matching_indices:
        sample = dataset[matching_indices[0]]
        ground_truth = sample["labels"].numpy()
        ground_truth_image = colorize_mask(ground_truth)
    else:
        ground_truth_image = Image.new(
            "RGB",
            resized.size,
            (0, 0, 0),
        )

    # Build a side-by-side comparison.
    panel_width, panel_height = resized.size
    comparison = Image.new(
        "RGB",
        (panel_width * 3, panel_height + 45),
        (30, 30, 30),
    )

    comparison.paste(resized, (0, 45))
    comparison.paste(ground_truth_image, (panel_width, 45))
    comparison.paste(prediction_image, (panel_width * 2, 45))

    from PIL import ImageDraw

    draw = ImageDraw.Draw(comparison)
    draw.text((10, 12), "Original Image", fill="white")
    draw.text((panel_width + 10, 12), "Ground Truth", fill="white")
    draw.text((panel_width * 2 + 10, 12), "Prediction", fill="white")

    output_path = OUTPUT_DIR / f"{image_path.stem}_comparison.png"
    comparison.save(output_path)

    print("\nInference completed.")
    print("Predicted classes:")
    for class_id in np.unique(prediction):
        if class_id < len(TRAINABLE_CLASSES):
            print(
                f"- {TRAINABLE_CLASSES[class_id]}: "
                f"{int((prediction == class_id).sum())} pixels"
            )

    print("\nSaved comparison to:")
    print(output_path)


if __name__ == "__main__":
    main()
