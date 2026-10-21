
from transformers import SegformerForSemanticSegmentation

from src.dataset import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
)


# Pretrained SegFormer model with road-scene knowledge.
PRETRAINED_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"


def get_trainable_classes():
    """Return class names excluding the unlabelled class."""
    return [
        name
        for name in CLASS_NAMES
        if name.lower() != "unlabelled"
    ]


def build_model():
    """
    Load pretrained SegFormer-B0 and configure it for CamVid.

    The pretrained segmentation head is replaced because CamVid
    has 11 trainable classes rather than the source dataset's classes.
    """
    class_names = get_trainable_classes()
    num_labels = len(class_names)

    id2label = {
        index: name
        for index, name in enumerate(class_names)
    }

    label2id = {
        name: index
        for index, name in id2label.items()
    }

    print("Loading pretrained SegFormer-B0...")
    print("Checkpoint:", PRETRAINED_MODEL)
    print("Number of trainable classes:", num_labels)
    print("Ignore index:", IGNORE_INDEX)

    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Ignore unlabelled pixels when calculating segmentation loss.
    model.config.semantic_loss_ignore_index = IGNORE_INDEX

    model.eval()

    print("\nModel loaded successfully.")
    print("Model type:", model.__class__.__name__)
    print("Number of output classes:", model.config.num_labels)
    print(
        "Model ignore index:",
        model.config.semantic_loss_ignore_index,
    )

    return model


if __name__ == "__main__":
    model = build_model()

    print("\nModel verification completed.")
