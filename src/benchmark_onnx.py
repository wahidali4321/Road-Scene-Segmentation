
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.model import build_model


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "models" / "best_segformer_camvid.pth"
ONNX_PATH = ROOT / "models" / "segformer_camvid.onnx"
TEST_DIR = ROOT / "data" / "test"

IMAGE_SIZE = (360, 480)
WARMUP_RUNS = 3
BENCHMARK_RUNS = 10


def main():
    if not CHECKPOINT.exists() or not ONNX_PATH.exists():
        raise FileNotFoundError("Checkpoint or ONNX model is missing.")

    torch.set_num_threads(1)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_paths = sorted(
        p for p in TEST_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {TEST_DIR}")

    image = Image.open(image_paths[0]).convert("RGB")
    image = image.resize(
        (IMAGE_SIZE[1], IMAGE_SIZE[0]),
        Image.Resampling.BILINEAR,
    )

    x = to_tensor(image).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    x_np = x.numpy().astype(np.float32)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(ONNX_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def pytorch_inference():
        with torch.inference_mode():
            return model(pixel_values=x).logits

    def onnx_inference():
        return session.run([output_name], {input_name: x_np})[0]

    # Warm up both models before timing.
    print("Warming up models...")
    for _ in range(WARMUP_RUNS):
        pytorch_inference()
        onnx_inference()

    pytorch_times = []
    onnx_times = []

    print(f"Benchmarking {BENCHMARK_RUNS} runs per model...")

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()
        pytorch_inference()
        pytorch_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        onnx_inference()
        onnx_times.append(time.perf_counter() - start)

    pytorch_ms = np.mean(pytorch_times) * 1000
    onnx_ms = np.mean(onnx_times) * 1000
    speedup = pytorch_ms / onnx_ms if onnx_ms > 0 else float("inf")

    print("\n" + "=" * 42)
    print("CPU INFERENCE BENCHMARK")
    print("=" * 42)
    print(f"Image: {image_paths[0].name}")
    print(f"PyTorch average: {pytorch_ms:.2f} ms/image")
    print(f"ONNX average:    {onnx_ms:.2f} ms/image")
    print(f"PyTorch FPS:     {1000 / pytorch_ms:.2f}")
    print(f"ONNX FPS:        {1000 / onnx_ms:.2f}")
    print(f"ONNX speedup:    {speedup:.2f}x")

    if speedup > 1:
        print("ONNX was faster in this benchmark.")
    else:
        print("ONNX was not faster in this benchmark.")

    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
