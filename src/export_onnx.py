
from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_segformer_camvid.pth"
ONNX_PATH = PROJECT_ROOT / "models" / "segformer_camvid.onnx"

DEVICE = torch.device("cpu")
INPUT_SHAPE = (1, 3, 360, 480)


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Trained checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("Loading trained SegFormer-B0 model...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    id2label = {
        index: name for index, name in enumerate(class_names)
    }
    label2id = {
        name: index for index, name in id2label.items()
    }

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # Export only the model logits, not the training loss.
    class SegmentationWrapper(torch.nn.Module):
        def __init__(self, segmentation_model):
            super().__init__()
            self.model = segmentation_model

        def forward(self, pixel_values):
            return self.model(
                pixel_values=pixel_values
            ).logits

    export_model = SegmentationWrapper(model).eval()
    dummy_input = torch.randn(*INPUT_SHAPE, device=DEVICE)

    print("Exporting to ONNX...")
    print("Input shape:", INPUT_SHAPE)

    torch.onnx.export(
        export_model,
        (dummy_input,),
        str(ONNX_PATH),
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
        external_data=True,
    )

    print("\nONNX export completed.")
    print("Saved model:", ONNX_PATH)

    # Check that the exported model is structurally valid.
    import onnx

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed.")


if __name__ == "__main__":
    main()
