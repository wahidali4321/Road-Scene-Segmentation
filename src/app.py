
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "segformer_camvid.onnx"

# Model input dimensions
IMAGE_WIDTH = 480
IMAGE_HEIGHT = 360

# ImageNet normalization values
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# CamVid class colors, in dataset class-ID order
CLASS_NAMES = [
    "Sky",
    "Building",
    "Pole",
    "Road",
    "Sidewalk",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
]

CLASS_COLORS = np.array(
    [
        [128, 128, 128],  # Sky
        [128, 0, 0],      # Building
        [192, 192, 128],  # Pole
        [128, 64, 128],   # Road
        [0, 0, 192],      # Sidewalk
        [128, 128, 0],    # Tree
        [192, 128, 128],  # SignSymbol
        [64, 64, 128],    # Fence
        [64, 0, 128],     # Car
        [64, 64, 0],      # Pedestrian
        [0, 128, 192],    # Bicyclist
    ],
    dtype=np.uint8,
)


# Load the ONNX model once when the app starts
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ONNX model not found: {MODEL_PATH}\n"
        "Please make sure models/segformer_camvid.onnx exists."
    )

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("ONNX model loaded successfully.")
print(f"Model: {MODEL_PATH}")
print("Backend: ONNX Runtime CPU")


def predict(image):
    """Return the original image, segmentation mask, and overlay."""

    if image is None:
        raise gr.Error("Please upload a road-scene image first.")

    # Convert the uploaded image to RGB
    original = image.convert("RGB")
    original = original.resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        Image.Resampling.BILINEAR,
    )
    original_array = np.array(original, dtype=np.uint8)

    # Prepare the image for SegFormer
    image_float = original_array.astype(np.float32) / 255.0
    image_float = (image_float - MEAN) / STD

    # HWC -> CHW -> NCHW
    input_tensor = np.transpose(image_float, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

    # Run ONNX inference
    logits = session.run(
        [output_name],
        {input_name: input_tensor},
    )[0]

    # Resize each class's logits to the input image dimensions
    class_logits = [
        cv2.resize(
            logits[0, class_id],
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
        for class_id in range(len(CLASS_NAMES))
    ]

    class_logits = np.stack(class_logits, axis=0)

    # Select the class with the highest score at each pixel
    predicted_mask = np.argmax(class_logits, axis=0)

    # Convert class IDs to RGB colors
    colored_mask = CLASS_COLORS[predicted_mask]

    # Blend the mask with the original image
    overlay = cv2.addWeighted(
        original_array,
        0.55,
        colored_mask,
        0.45,
        0,
    )

    return original_array, colored_mask, overlay


# Build the Gradio interface
with gr.Blocks(title="Road Scene Segmentation") as demo:
    gr.Markdown(
        """
        # Road Scene Segmentation
        ### SegFormer-B0 + ONNX Runtime

        Upload a road-scene image to identify roads, buildings, cars,
        pedestrians, trees, and other CamVid classes.
        """
    )

    input_image = gr.Image(
        type="pil",
        label="Upload Road Image",
    )

    segment_button = gr.Button(
        "Segment Image",
        variant="primary",
    )

    with gr.Row():
        original_output = gr.Image(label="1. Original Image")
        mask_output = gr.Image(label="2. Segmentation Mask")
        overlay_output = gr.Image(label="3. Segmentation Overlay")

    segment_button.click(
        fn=predict,
        inputs=input_image,
        outputs=[
            original_output,
            mask_output,
            overlay_output,
        ],
    )

    gr.Markdown(
        """
        **Model:** SegFormer-B0 fine-tuned on CamVid  
        **Inference:** ONNX Runtime on CPU  
        **Output classes:** 11 trainable classes
        """
    )


if __name__ == "__main__":
    demo.launch()
