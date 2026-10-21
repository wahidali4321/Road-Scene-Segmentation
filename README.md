# Road-Scene-Segmentation
A deep learning project for semantic segmentation of road scenes using SegFormer-B0, trained on the CamVid dataset. The project supports PyTorch training and evaluation, ONNX model export, CPU inference benchmarking, and interactive image segmentation through a Gradio web application.

Project Overview

Semantic segmentation assigns a class label to each pixel in an image. In autonomous driving and road-scene understanding, this helps identify roads, buildings, vehicles, pedestrians, trees, and other elements in the environment.

This project uses a pretrained SegFormer-B0 model fine-tuned on CamVid to classify road-scene pixels into 11 trainable classes, excluding the unlabelled class.

Features

Fine-tuning SegFormer-B0 on the CamVid dataset.

Training, validation, and test evaluation.

Per-class Intersection over Union (IoU) analysis.

Confusion matrix generation.

PyTorch model inference.

Export to ONNX format.

ONNX Runtime CPU inference.

PyTorch versus ONNX performance benchmarking.

Interactive Gradio application for image segmentation.

Visualization of original images, segmentation masks, and overlays.

Technologies Used

Python

PyTorch

Hugging Face Transformers

SegFormer-B0

OpenCV

NumPy

ONNX

ONNX Runtime

Gradio

Matplotlib

Pandas

pytest

Dataset

This project uses the CamVid road-scene segmentation dataset, which contains road-scene images and their corresponding pixel-level segmentation masks.

Dataset split:

Split

Images

Training

367

Validation

101

Testing

233

Total

701

The original dataset contains 12 classes. The unlabelled class is ignored during training and evaluation, leaving 11 trainable classes.

Note: The dataset is not included in this GitHub repository. Download and arrange the CamVid dataset according to the paths expected by src/dataset.py.

Model Architecture

SegFormer-B0 is a transformer-based semantic segmentation architecture designed to extract multi-scale visual features efficiently.

The model was initialized using pretrained SegFormer-B0 weights and fine-tuned for the CamVid dataset.

Architecture: SegFormer-B0

Dataset: CamVid

Trainable classes: 11

Training epochs: 5

Input resolution: 512 × 512 (resized by the project preprocessing pipeline)

Project Structure

Road-Scene-Segmentation/
├── models/
│   ├── best_segformer_camvid.pth
│   └── segformer_camvid.onnx
├── outputs/
│   ├── predictions/
│   ├── test_class_iou.csv
│   └── test_confusion_matrix.csv
├── src/
│   ├── app.py
│   ├── benchmark_onnx.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   ├── inference.py
│   ├── model.py
│   ├── test_onnx.py
│   └── train.py
├── tests/
│   └── test_dataset.py
├── .gitignore
├── requirements.txt
└── README.md

Installation

1. Clone the repository

git clone https://github.com/wahidali4321/Road-Scene-Segmentation.git
cd Road-Scene-Segmentation

2. Create a virtual environment

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

If PowerShell blocks activation, use:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

3. Install dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

4. Prepare the dataset

Download the CamVid dataset and place its images and segmentation masks in the directory structure expected by src/dataset.py. Verify the dataset paths and configuration before training.

Training

To train the model:

python -m src.train

The training script saves the best-performing checkpoint to:

models/best_segformer_camvid.pth

Check src/train.py for the configured training parameters and dataset paths.

Evaluation

Evaluate the trained model on the test dataset:

python -m src.evaluate

Evaluation includes metrics such as:

Mean Intersection over Union (mIoU)

Pixel accuracy

Per-class IoU

Confusion matrix

Generated evaluation results are stored in the outputs/ directory.

ONNX Export and Inference

Export the trained model

python -m src.export_onnx

The exported model is saved as:

models/segformer_camvid.onnx

Test ONNX inference

python -m src.test_onnx

This script checks ONNX inference and compares its output with the PyTorch model.

Benchmark performance

python -m src.benchmark_onnx

The benchmark compares average inference latency and throughput between PyTorch and ONNX Runtime on the available hardware.

Run the Gradio Application

Start the interactive segmentation application:

python -m src.app

Open the local URL displayed in the terminal, typically:

http://127.0.0.1:7860

Upload a road-scene image and click Segment Image to generate:

Original image

Color-coded semantic segmentation mask

Overlay combining the original image and segmentation mask

The application uses the exported ONNX model for CPU inference.

Results

Model Evaluation

Metric

Result

Best validation mIoU

50.93%

Test mIoU

47.47%

Test pixel accuracy

89.06%

ONNX Performance Benchmark

Metric

PyTorch

ONNX Runtime

Average latency

505.65 ms

278.23 ms

Approximate throughput

1.98 FPS

3.59 FPS

The ONNX model achieved approximately 1.82× faster inference in the recorded single-image CPU benchmark, with around 44.98% lower latency.

These measurements represent the tested hardware and configuration. Performance may vary across systems and input sizes.

Example Prediction

A sample comparison image is available at:

outputs/predictions/0001TP_008550_comparison.png

Additional evaluation results are available in the CSV files under outputs/.

Testing

Run the available unit tests:

python -m pytest tests/ -v

Limitations

Performance depends on hardware and input image resolution.

Some visually small or infrequent classes, such as poles and bicyclists, achieved low IoU on the test set.

Results depend on dataset quality, class distribution, and training configuration.

The benchmark reports single-image inference and does not establish real-time performance under every deployment condition.

The CamVid dataset is not distributed with this repository.

Future Improvements

Train for more epochs and experiment with hyperparameters.

Improve performance on underrepresented classes.

Evaluate data augmentation and class-weighted loss functions.

Test larger SegFormer variants.

Explore GPU and edge-device deployment.

Improve inference speed and application usability.

License

Add a suitable license before redistributing or reusing this project. Also verify the licensing terms of the pretrained model and CamVid dataset.

Author

Wahid Ali

GitHub: https://github.com/wahidali4321
