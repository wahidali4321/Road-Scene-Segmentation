
import unittest

import torch

from src.dataset import (
    CamVidDataset,
    CLASS_NAMES,
    COLOR_TO_ID,
    NUM_CLASSES,
)


class TestCamVidDataset(unittest.TestCase):

    def test_palette(self):
        self.assertEqual(NUM_CLASSES, 12)
        self.assertEqual(len(CLASS_NAMES), 12)
        self.assertEqual(COLOR_TO_ID[(128, 64, 128)], 3)

    def test_training_dataset(self):
        dataset = CamVidDataset("train")
        self.assertEqual(len(dataset), 367)

    def test_sample_shapes_and_types(self):
        dataset = CamVidDataset("train")
        sample = dataset[0]

        self.assertEqual(
            tuple(sample["pixel_values"].shape), (3, 360, 480)
        )
        self.assertEqual(
            tuple(sample["labels"].shape), (360, 480)
        )
        self.assertEqual(sample["labels"].dtype, torch.int64)

    def test_mask_labels_are_valid(self):
        dataset = CamVidDataset("train")
        labels = dataset[0]["labels"]

        valid = (labels == 255) | (
            (labels >= 0) & (labels < NUM_CLASSES)
        )
        self.assertTrue(torch.all(valid).item())

    def test_invalid_split(self):
        with self.assertRaises(ValueError):
            CamVidDataset("invalid")


if __name__ == "__main__":
    unittest.main()
