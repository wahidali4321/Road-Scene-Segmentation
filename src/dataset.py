
from pathlib import Path
import csv

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# 1. Project paths and configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PALETTE_FILE = DATA_DIR / "class_palette.csv"

IMAGE_DIRS = {
    "train": DATA_DIR / "train",
    "val": DATA_DIR / "val",
    "test": DATA_DIR / "test",
}

MASK_DIRS = {
    "train": DATA_DIR / "trainannot",
    "val": DATA_DIR / "valannot",
    "test": DATA_DIR / "testannot",
}

IGNORE_INDEX = 255


# ============================================================
# 2. Load RGB class palette
# ============================================================

def load_palette(palette_path=PALETTE_FILE):
    """
    Read class names and RGB colors from class_palette.csv.

    Returns:
        class_names: List of class names.
        color_to_id: Dictionary mapping RGB tuples to class IDs.
    """
    class_names = []
    color_to_id = {}

    if not Path(palette_path).is_file():
        raise FileNotFoundError(
            f"Class palette not found: {palette_path}"
        )

    with open(
        palette_path,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {"name", "r", "g", "b"}

        if not reader.fieldnames or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "class_palette.csv must contain columns: name,r,g,b"
            )

        for class_id, row in enumerate(reader):
            name = row["name"].strip()

            color = (
                int(row["r"]),
                int(row["g"]),
                int(row["b"]),
            )

            if color in color_to_id:
                raise ValueError(
                    f"Duplicate RGB color in palette: {color}"
                )

            class_names.append(name)
            color_to_id[color] = class_id

    if not class_names:
        raise ValueError("The class palette contains no classes.")

    return class_names, color_to_id


CLASS_NAMES, COLOR_TO_ID = load_palette()
NUM_CLASSES = len(CLASS_NAMES)

# Find the unlabelled class ID by its palette entry.
UNLABELLED_ID = CLASS_NAMES.index("unlabelled") if (
    "unlabelled" in CLASS_NAMES
) else None


# ============================================================
# 3. CamVid dataset
# ============================================================

class CamVidDataset(Dataset):
    """
    CamVid semantic segmentation dataset.

    Expected directory structure:

        data/
            class_palette.csv
            train/
            trainannot/
            val/
            valannot/
            test/
            testannot/

    Images:
        float32 tensors with shape [3, H, W], values in [0, 1].

    Masks:
        int64 tensors with shape [H, W].
        Valid class IDs are retained.
        Unlabelled pixels and unknown colors use IGNORE_INDEX.
    """

    def __init__(
        self,
        split="train",
        image_size=(360, 480),
    ):
        if split not in IMAGE_DIRS:
            raise ValueError(
                f"Invalid split '{split}'. "
                "Choose 'train', 'val', or 'test'."
            )

        if (
            not isinstance(image_size, (tuple, list))
            or len(image_size) != 2
            or not all(
                isinstance(value, int) and value > 0
                for value in image_size
            )
        ):
            raise ValueError(
                "image_size must be positive integers: (height, width)"
            )

        self.split = split
        self.image_dir = IMAGE_DIRS[split]
        self.mask_dir = MASK_DIRS[split]
        self.image_size = tuple(image_size)

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        if not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"Mask directory not found: {self.mask_dir}"
            )

        self.samples = []
        self.unknown_colors = set()

        image_paths = sorted(self.image_dir.glob("*.png"))

        # Match 0001TP_006690.png with 0001TP_006690_L.png.
        for image_path in image_paths:
            mask_path = self.mask_dir / f"{image_path.stem}_L.png"

            if mask_path.is_file():
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError(
                f"No matching image-mask pairs found in '{split}'. "
                f"Images: {self.image_dir}, masks: {self.mask_dir}"
            )

        missing_masks = len(image_paths) - len(self.samples)

        if missing_masks:
            print(
                f"Warning: {missing_masks} images in '{split}' "
                "have no matching mask."
            )

    def __len__(self):
        """Return the number of image-mask pairs."""
        return len(self.samples)

    def _mask_to_ids(self, mask):
        """
        Convert an RGB segmentation mask into class IDs.

        Known colors map to their palette class IDs.
        Unknown colors and the unlabelled class map to IGNORE_INDEX.
        """
        rgb = np.asarray(
            mask.convert("RGB"),
            dtype=np.uint8,
        )

        # Initially ignore all pixels until a palette color matches.
        label_mask = np.full(
            rgb.shape[:2],
            fill_value=IGNORE_INDEX,
            dtype=np.uint8,
        )

        # Map RGB colors to their corresponding class IDs.
        for color, class_id in COLOR_TO_ID.items():
            matches = np.all(rgb == color, axis=-1)
            label_mask[matches] = class_id

        # Track genuinely unknown RGB colors for debugging.
        unknown_pixels = label_mask == IGNORE_INDEX

        if unknown_pixels.any():
            unknown_rgb = np.unique(
                rgb[unknown_pixels].reshape(-1, 3),
                axis=0,
            )

            for color in unknown_rgb:
                self.unknown_colors.add(
                    tuple(int(value) for value in color)
                )

        # Convert only the unlabelled class to IGNORE_INDEX.
        if UNLABELLED_ID is not None:
            label_mask[label_mask == UNLABELLED_ID] = IGNORE_INDEX

        return label_mask

    def __getitem__(self, index):
        """Load and preprocess one image-mask pair."""
        image_path, mask_path = self.samples[index]

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        with Image.open(mask_path) as source:
            mask = source.convert("RGB")

        if image.size != mask.size:
            raise ValueError(
                f"Image-mask size mismatch for {image_path.name}: "
                f"image={image.size}, mask={mask.size}"
            )

        height, width = self.image_size
        target_size = (width, height)

        # Resize the image smoothly.
        image = image.resize(
            target_size,
            Image.Resampling.BILINEAR,
        )

        # Nearest-neighbor resizing preserves discrete mask colors.
        mask = mask.resize(
            target_size,
            Image.Resampling.NEAREST,
        )

        # Convert image to float32 [3, H, W], range [0, 1].
        image_array = (
            np.asarray(image, dtype=np.float32) / 255.0
        )

        image_array = np.ascontiguousarray(
            image_array.transpose(2, 0, 1)
        )

        image_tensor = torch.from_numpy(image_array)

        # Convert RGB mask into integer labels [H, W].
        label_array = self._mask_to_ids(mask)
        label_array = np.ascontiguousarray(
            label_array.astype(np.int64)
        )

        label_tensor = torch.from_numpy(label_array)

        return {
            "pixel_values": image_tensor,
            "labels": label_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


# ============================================================
# 4. Dataset verification
# ============================================================

if __name__ == "__main__":
    print("CamVid dataset verification")
    print("=" * 45)
    print("Number of palette classes:", NUM_CLASSES)
    print("Class names:", CLASS_NAMES)
    print("Unlabelled class ID:", UNLABELLED_ID)
    print("Ignore index:", IGNORE_INDEX)

    for split in ("train", "val", "test"):
        dataset = CamVidDataset(split=split)

        print(f"\n{split.upper()} samples: {len(dataset)}")

        sample = dataset[0]
        image = sample["pixel_values"]
        labels = sample["labels"]

        print("Image shape:", tuple(image.shape))
        print("Mask shape:", tuple(labels.shape))
        print("Image dtype:", image.dtype)
        print("Mask dtype:", labels.dtype)
        print("Unique labels:", torch.unique(labels).tolist())
        print("Valid pixels:", (labels != IGNORE_INDEX).sum().item())
        print("Ignored pixels:", (labels == IGNORE_INDEX).sum().item())

    print("\nDataset verification completed.")
