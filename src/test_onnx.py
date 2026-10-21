
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_segformer_camvid.pth"
ONNX_PATH = PROJECT_ROOT / "models" / "segformer_camvid.onnx"
TEST_DIR = PROJECT_ROOT / "data" / "test"

DEVICE = torch.device("cpu")
IMAGE_SIZE = (360, 480)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)
    if not ONNX_PATH.exists():
        raise FileNotFoundError(ONNX_PATH)

    images = sorted(
        path for path in TEST_DIR.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not images:
        raise FileNotFoundError(f"No test images found in {TEST_DIR}")

    image_path = images[0]
    print("Test image:", image_path.name)

    image = Image.open(image_path).convert("RGB")
    image = image.resize(
        (IMAGE_SIZE[1], IMAGE_SIZE[0]),
        Image.Resampling.BILINEAR,
    )

    input_tensor = to_tensor(image).unsqueeze(0)
    input_tensor = (input_tensor - MEAN) / STD
    input_array = input_tensor.numpy().astype(np.float32)

    # Load trained PyTorch model and restore its checkpoint.
    print("\nLoading PyTorch model...")
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    pytorch_model = build_model()
    pytorch_model.load_state_dict(checkpoint["model_state_dict"])
    pytorch_model.eval()

    with torch.inference_mode():
        pytorch_logits = pytorch_model(
            pixel_values=torch.from_numpy(input_array)
        ).logits

        pytorch_logits = F.interpolate(
            pytorch_logits,
            size=IMAGE_SIZE,
            mode="bilinear",
            align_corners=False,
        ).numpy()

    # Load exported ONNX model.
    print("Loading ONNX Runtime model...")
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    onnx_logits = session.run(
        [output_name],
        {input_name: input_array},
    )[0]

    # Compare raw logits before resizing.
    pytorch_raw = pytorch_model_output_raw(
        pytorch_model, input_array
    )
    print("PyTorch raw output shape:", pytorch_raw.shape)
    print("ONNX raw output shape:", onnx_logits.shape)

    max_difference = float(
        np.max(np.abs(pytorch_raw - onnx_logits))
    )
    mean_difference = float(
        np.mean(np.abs(pytorch_raw - onnx_logits))
    )

    pytorch_predictions = np.argmax(pytorch_logits, axis=1)
    onnx_resized = F.interpolate(
        torch.from_numpy(onnx_logits),
        size=IMAGE_SIZE,
        mode="bilinear",
        align_corners=False,
    ).numpy()
    onnx_predictions = np.argmax(onnx_resized, axis=1)

    agreement = float(
        np.mean(pytorch_predictions == onnx_predictions) * 100
    )

    print("\n" + "=" * 42)
    print("PYTORCH vs ONNX COMPARISON")
    print("=" * 42)
    print(f"Maximum absolute logit difference: {max_difference:.6f}")
    print(f"Mean absolute logit difference:    {mean_difference:.6f}")
    print(f"Pixel prediction agreement:        {agreement:.2f}%")

    if agreement >= 99:
        print("\nExcellent: predictions agree very closely.")
    elif agreement >= 95:
        print("\nGood: predictions show strong agreement.")
    else:
        print("\nReview the differences before deploying ONNX.")

    print("\nONNX inference test completed.")


def pytorch_model_output_raw(model, input_array):
    with torch.inference_mode():
        output = model(
            pixel_values=torch.from_numpy(input_array)
        ).logits
    return output.numpy()


if __name__ == "__main__":
    main()
