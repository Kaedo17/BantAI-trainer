# YOLO Training Pipeline

A complete, user-friendly pipeline for training YOLO object detection models with automatic dataset structure detection, augmentation, splitting, merging, and testing. Built for RTX 50-series GPUs with full CUDA support.

Python 3.10+ | PyTorch 2.0+ | Ultralytics 8.0+ | MIT License

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Supported Dataset Structures](#supported-dataset-structures)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)
- [Output Structure](#output-structure)
- [FAQ](#faq)

---

## Overview

This pipeline provides a complete workflow for training YOLO models (YOLOv8, YOLOv9, YOLOv10, and the latest YOLO26). It automatically detects your dataset folder structure, handles augmentation, splits data into train/val/test sets, merges multiple datasets, trains the model, and allows you to test it with adjustable confidence thresholds.

Whether you're a beginner or an expert, this pipeline simplifies the entire process from raw images to a trained model ready for deployment.

---

## Features

| Feature | Description |
|---------|-------------|
| Auto Structure Detection | Automatically detects 4 different YOLO folder structures |
| Image Augmentation | 7 augmentation types (flip, dark, bright, gray, blur, contrast) |
| Dataset Splitting | 80/10/10 train/val/test split with random shuffling |
| Dataset Merging | Merge multiple datasets with automatic class mapping |
| Model Training | Supports YOLOv8, YOLOv9, YOLOv10, YOLO26 |
| Auto Model Download | Automatically downloads missing model files |
| GPU Support | Full CUDA support with automatic GPU detection (RTX 50-series ready) |
| Model Testing | Test with adjustable confidence and IOU thresholds |
| Multiple Sources | Test on images, folders, videos, webcam, or URLs |
| Interactive Menu | Easy-to-use menu system with navigation (back/cancel options) |
| Organized Output | All outputs saved in structured folders (Output/, Trained/) |
| Class Preservation | Maintains class information through all pipeline steps |

---

## Folder Structure

BantAI trainer/
├── main.py                 # Main controller with interactive menu
├── merge.py                # Dataset merging script
├── augment.py              # Image augmentation script
├── split.py                # Dataset splitting script
├── train.py                # Model training script
├── test.py                 # Model testing script
├── config.py               # Configuration settings
├── utils.py                # Utility functions
├── requirements.txt        # Python dependencies
│
├── Output/                 # All processed datasets
│   ├── Merged/             # Merged datasets
│   ├── Augmented/          # Augmented datasets
│   └── Split/              # Split datasets (train/val/test)
│
└── Trained/                # Trained models
    └── [model_name]/       # Your trained model folder
        ├── weights/
        │   ├── best.pt     # Best model weights
        │   └── last.pt     # Final epoch weights
        ├── results.png     # Training graphs
        ├── results.csv     # Raw training metrics
        └── args.yaml       # Training arguments

---

## Installation

### Prerequisites

- Python: 3.10 - 3.12 (3.14 may have compatibility issues)
- GPU: NVIDIA GPU with CUDA support (RTX 5050 recommended)
- RAM: 8GB minimum (16GB recommended)
- Storage: 10GB+ free space for models and datasets

### Step 1: Download the Pipeline

Create a project folder and download all Python files into it.

### Step 2: Create a Virtual Environment (Recommended)

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

### Step 3: Install Dependencies

pip install -r requirements.txt

### Step 4: Install PyTorch with CUDA (for RTX 50-series)

# For RTX 50 series with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

---

## Quick Start

### Run the Pipeline

python main.py

### Main Menu

============================================================
  YOLO Training Pipeline
============================================================

Available Options:
   ┌─────────────────────────────────────────────────┐
   │ 0.  Merge Multiple Datasets (NEW!)              │
   │ 1.  Augment Images                              │
   │ 2.  Split Dataset (80% Train, 10% Val, 10% Test)│
   │ 3.  Train Model                                 │
   │ 4.  Test Model                                  │
   │ 5.  Complete Pipeline (Merge → Augment → Split → Train) │
   │ 6.  Quick Train (Skip Augment/Split)            │
   │ 7.  View GPU Info                               │
   │ 8.  Exit                                        │
   └─────────────────────────────────────────────────┘

👉 Select an option (0-8):

### Typical Workflow

1. Option 0 (Merge) - Combine multiple datasets (optional)
2. Option 1 (Augment) - Expand your dataset with variations
3. Option 2 (Split) - Create train/val/test splits
4. Option 3 (Train) - Train YOLO model
5. Option 4 (Test) - Evaluate model performance

Or use Option 5 for the full automated workflow!

---

## Detailed Usage

### Option 0: Merge Multiple Datasets

Combine multiple YOLO datasets into one unified dataset with automatic class mapping.

What it does:
- Automatically detects folder structures
- Maps class IDs across datasets (matching by class name)
- Updates all label files with new global class IDs
- Creates unique filenames to avoid conflicts
- Outputs merged dataset to Output/Merged/

Supported inputs:
- Any number of datasets (2, 3, 5, etc.)
- Any YOLO folder structure
- Datasets with different class counts

Example:
Dataset 1: OpenFlameHazards (class 0: "Open-Flame-Hazard")
Dataset 2: HeavyWoodenFurniture (class 0: "Heavy-Wooden-Furniture")
Merged result: class 0: "Open-Flame-Hazard", class 1: "Heavy-Wooden-Furniture"

---

### Option 1: Augment Images

Creates variations of your training images to increase dataset size and improve model robustness.

Augmentations:
- orig - Original image (always included)
- flipH - Horizontal flip (mirror left-right)
- flipV - Vertical flip (mirror up-down)
- dark - Darker image (alpha 0.4)
- bright - Brighter image (alpha 1.6, beta 30)
- gray - Grayscale conversion
- blur - Gaussian blur (5x5 kernel)
- contrast - Enhanced contrast (alpha 1.8, beta -30)

Features:
- Preserves existing val/test splits
- Copies data.yaml to preserve class information
- Outputs to Output/Augmented/

---

### Option 2: Split Dataset

Splits your dataset into train/val/test sets for proper model evaluation.

Split Ratios:
- Train: 80% - Used for model training
- Val: 10% - Used for validation during training
- Test: 10% - Used for final evaluation

Features:
- Automatically detects folder structure
- Preserves class information from data.yaml
- Handles datasets with only training images
- Outputs to Output/Split/

---

### Option 3: Train Model

Trains a YOLO model on your dataset with automatic model download.

Supported Models:
- yolo26s.pt - YOLO26 small (Recommended - NMS-free, 43% faster CPU)
- yolo26n.pt - YOLO26 nano (Edge devices, mobile)
- yolov10n.pt - YOLOv10 nano (Fast inference)
- yolov8s.pt - YOLOv8 small (Balanced performance)
- yolov9t.pt - YOLOv9 tiny (Ultra-lightweight)

Training Parameters:
- Epochs: 150 - Training cycles through the dataset
- Batch Size: 16 - Images processed together
- Image Size: 640 - Resize images to this dimension
- Workers: 2 - Parallel data loading threads

Output:
- Model saved to Trained/[model_name]/weights/best.pt
- Training metrics saved to Trained/[model_name]/results.csv
- Training graphs saved to Trained/[model_name]/results.png

---

### Option 4: Test Model

Evaluates your trained model on new data with adjustable confidence threshold.

Testing Options:
1. Single image file - Test one image
2. Folder of images - Batch test multiple images
3. Video file - Test on video
4. Webcam - Real-time detection
5. Image URL - Test image from URL

Adjustable Settings:
- Confidence threshold (0-1): Minimum confidence to show detections
- IOU threshold (0-1): NMS overlap threshold

Model Selection:
- Automatically browses Trained/ folder for models
- Option to enter custom path

---

## Supported Dataset Structures

The pipeline automatically detects these folder structures:

### Structure 1: Standard YOLO Split

your_dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    ├── val/
    └── test/

### Structure 2: Alternative Split

your_dataset/
├── train/
│   ├── images/
│   │   └── image1.jpg
│   └── labels/
│       └── image1.txt
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

### Structure 3: Flat (Unsplitted)

your_dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...

### Structure 4: Mixed (Images + Labels together)

your_dataset/
├── image1.jpg
├── image1.txt
├── image2.jpg
├── image2.txt
└── ...

---

## Configuration

### config.py - Main Settings

# Model settings
DEFAULT_MODEL = "yolo26s.pt"  # Default model
AVAILABLE_MODELS = [...]       # All supported models

# Training defaults
DEFAULT_EPOCHS = 150
DEFAULT_BATCH = 16
DEFAULT_IMGSZ = 640
DEFAULT_WORKERS = 2

# Augmentation
DEFAULT_AUG_TARGET = 2000  # Max augmented images

# Split ratios
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # 80/10/10 split

### Model URLs

All model files are automatically downloaded from the official Ultralytics assets repository:

- yolo26s.pt: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt
- yolov8s.pt: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt
- yolov10n.pt: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10n.pt

---

## Troubleshooting

Common Issues and Solutions:

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA error: no kernel image available | PyTorch version doesn't support RTX 5050 | Install PyTorch with CUDA 12.8 |
| Label class exceeds dataset class count | Class IDs don't match data.yaml | Run split script or update data.yaml |
| No images with matching labels found | Labels missing or wrong structure | Ensure labels in labels/ folder with same names as images |
| val: Error loading data from ...\images\val | No validation images | Run split script to create val/test splits |
| ModuleNotFoundError: No module named 'executorch' | ExecuTorch not installed | Skip export or use ONNX/TFLite |
| Model download fails (404) | Wrong model name | Use yolo26s.pt (not yolov26s.pt) |

Quick Fix Commands:

# Install PyTorch with CUDA 12.8 for RTX 50-series
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Fix dataset structure
python split.py

# Test with TFLite instead of ExecuTorch
model.export(format="tflite")

---

## Examples

### Example 1: Complete Pipeline from Scratch

1. Place your images in a folder with labels:
my_dataset/
├── images/
│   ├── fire1.jpg
│   └── ...
└── labels/
    ├── fire1.txt
    └── ...

2. Run the pipeline:
python main.py
Select option 5 (Complete Pipeline)

3. Follow prompts:
   - Merge datasets? (if multiple)
   - Enter input folder: my_dataset
   - Enter output folder name: (press Enter for default)
   - Set augmentation target: 3000
   - Set split folder name: (press Enter)
   - Select model: yolo26s.pt
   - Set epochs: 150
   - Enter model name: Fire_Detector_v1

4. Wait for training to complete

5. Test model with option 4

### Example 2: Merge Multiple Datasets

python main.py
Select option 0

Enter number of datasets: 2
Dataset 1: C:\datasets\open_flame
Dataset 2: C:\datasets\exposed_ceiling
Enter merged folder name: flame_detection_merged

Output: Output/Merged/flame_detection_merged/

### Example 3: Quick Train on Existing Dataset

python main.py
Select option 6 (Quick Train)

Enter dataset path: Output/Split/Master_Dataset_Aug_Split
Select model: yolo26s.pt
Enter model name: Quick_Model

### Example 4: Test with 70% Confidence

python main.py
Select option 4 (Test Model)

Select model from list or enter path
Enter confidence threshold: 0.7
Choose test option: 1 (Single image)
Enter image path: test_image.jpg

### Example 5: Command Line Mode

# Quick train
python main.py --mode quick

# Train with specific parameters
python main.py --mode train --dataset my_dataset --model yolo26s.pt --epochs 100 --batch 8

# Test with custom confidence
python main.py --mode test --conf 0.7

---

## Output Structure

### Training Output (Trained/[model_name]/)

Trained/Fire_Detector_v1/
├── weights/
│   ├── best.pt         # Best model (use for testing)
│   └── last.pt         # Final epoch model
├── results.png         # Training graphs (loss, mAP, precision, recall)
├── results.csv         # Raw training metrics
├── args.yaml           # Training arguments used
├── labels.jpg          # Label distribution visualization
└── train_batch*.jpg    # Sample training batches

### Processed Datasets (Output/)

Output/
├── Merged/
│   └── merged_dataset/
│       ├── images/train/
│       ├── labels/train/
│       └── data.yaml
├── Augmented/
│   └── dataset_augmented/
│       ├── images/train/
│       ├── images/val/
│       ├── images/test/
│       ├── labels/train/
│       ├── labels/val/
│       ├── labels/test/
│       └── data.yaml
└── Split/
    └── dataset_split/
        ├── images/train/
        ├── images/val/
        ├── images/test/
        ├── labels/train/
        ├── labels/val/
        ├── labels/test/
        └── data.yaml

---

## FAQ

Q: What's the difference between YOLO26 and YOLOv8?
A: YOLO26 is the latest NMS-free architecture with 43% faster CPU inference, no post-processing (NMS-free), MuSGD optimizer for better convergence, and streamlined architecture (no DFL).

Q: Can I train on CPU only?
A: Yes, but training will be significantly slower. The pipeline automatically falls back to CPU if no GPU is detected.

Q: How do I add custom classes?
A: The class names are automatically extracted from your labels and data.yaml. Just ensure your label files have the correct class IDs and your data.yaml lists all class names.

Q: What if I have more than 7 classes?
A: The pipeline supports any number of classes. Class mapping during merge automatically handles up to the maximum integer class ID.

Q: Can I resume interrupted training?
A: Yes, the training script saves checkpoints. To resume, use resume=True in the training parameters.

Q: Why is my val folder empty?
A: If your dataset only has training images, you need to run the split script to create validation and test splits.

Q: How do I export to Android?
A: Use TFLite format (more stable than ExecuTorch):
model.export(format="tflite")

Q: What do "back" and "cancel" options do?
A: "back" returns to the previous menu. "cancel" exits the current operation.

---

## License

This project is open-source and available under the MIT License.

---

## Acknowledgments

- Ultralytics for YOLO implementation
- PyTorch for deep learning framework
- OpenCV for image processing

---

Happy Training! 🎉