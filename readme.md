# 🚀 YOLO Training Pipeline

A complete, user-friendly pipeline for training YOLO object detection models with automatic dataset structure detection, augmentation, splitting, training, and testing.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Folder Structures Supported](#folder-structures-supported)
- [Detailed Usage](#detailed-usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

---

## 📖 Overview

This pipeline provides a complete workflow for training YOLO models (YOLOv8, YOLOv9, YOLOv10, and the latest **YOLO26**). It automatically detects your dataset folder structure, handles augmentation, splits data into train/val/test sets, trains the model, and allows you to test it with adjustable confidence thresholds.

Whether you're a beginner or an expert, this pipeline simplifies the entire process from raw images to a trained model ready for deployment.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Auto Structure Detection** | Automatically detects 4 different YOLO folder structures |
| 📸 **Image Augmentation** | 7 augmentation types (flip, dark, bright, gray, blur, contrast) |
| 📊 **Dataset Splitting** | 80/10/10 train/val/test split with random shuffling |
| 🎯 **Model Training** | Supports YOLOv8, YOLOv9, YOLOv10, YOLO26 |
| 💾 **Auto Model Download** | Automatically downloads missing model files |
| 🎮 **GPU Support** | Full CUDA support with automatic GPU detection |
| 🧪 **Model Testing** | Test with adjustable confidence and IOU thresholds |
| 📁 **Multiple Sources** | Test on images, folders, videos, webcam, or URLs |
| 🖥️ **Interactive Menu** | Easy-to-use menu system with command-line options |

---

## 🔧 Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (RTX 5050 recommended)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB+ free space for models and datasets

### Software
- **Python**: 3.10 - 3.12 (3.14 not fully supported yet)
- **CUDA**: 12.8 or higher for RTX 50 series
- **Operating System**: Windows 10/11, Linux, or macOS

---

## 📦 Installation

### Step 1: Clone or Download the Pipeline
```bash
# Create a project folder
mkdir yolo-pipeline
cd yolo-pipeline

# Download all Python files into this folder
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 4: Install PyTorch with CUDA (for GPU)
```bash
# For RTX 50 series with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## 🚀 Quick Start

### Run the Pipeline
```bash
python main.py
```

You'll see this menu:

```
============================================================
  YOLO Training Pipeline
============================================================

📋 Available Options:
   ┌─────────────────────────────────────────────────┐
   │ 1.  Augment Images                              │
   │ 2.  Split Dataset (80% Train, 10% Val, 10% Test)│
   │ 3.  Train Model                                 │
   │ 4.  Test Model                                  │
   │ 5.  Complete Pipeline (Augment → Split → Train) │
   │ 6.  Quick Train (Skip Augment/Split)            │
   │ 7.  View GPU Info                               │
   │ 8.  Exit                                        │
   └─────────────────────────────────────────────────┘

👉 Select an option (1-8):
```

### Typical Workflow

1. **Option 5** (Complete Pipeline) - if you have raw images and need everything
2. **Option 6** (Quick Train) - if your data is already split and organized
3. **Option 4** (Test Model) - after training to evaluate performance

---

## 📁 Folder Structures Supported

The pipeline automatically detects these folder structures:

### Structure 1: Standard YOLO Split
```
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
```

### Structure 2: Alternative Split
```
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
```

### Structure 3: Flat (Unsplitted)
```
your_dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
```

### Structure 4: Mixed (Images + Labels together)
```
your_dataset/
├── image1.jpg
├── image1.txt
├── image2.jpg
├── image2.txt
└── ...
```

---

## 📖 Detailed Usage

### 1. Augment Images
Expands your dataset by creating variations of existing images.

```bash
# Via menu: Option 1
```

**What it does:**
- Creates 8 variations per image (original + 7 augmentations)
- Preserves existing val/test splits
- Saves to a new folder with "_augmented" suffix

**Augmentations:**
- `flipH` - Horizontal flip
- `flipV` - Vertical flip
- `dark` - Darker image
- `bright` - Brighter image
- `gray` - Grayscale conversion
- `blur` - Gaussian blur
- `contrast` - Enhanced contrast

### 2. Split Dataset
Splits your dataset into train/val/test sets.

```bash
# Via menu: Option 2
```

**What it does:**
- Detects your folder structure automatically
- Splits images: 80% train, 10% val, 10% test
- Creates `data.yaml` configuration file
- Outputs to a new folder with "_split" suffix

### 3. Train Model
Trains a YOLO model on your dataset.

```bash
# Via menu: Option 3
```

**What it does:**
- Downloads the selected model automatically if missing
- Uses your GPU if available
- Shows training progress with mAP metrics
- Saves best and last model checkpoints

**Training Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| Epochs | 150 | Training cycles through the dataset |
| Batch Size | 16 | Images processed together |
| Image Size | 640 | Resize images to this dimension |
| Workers | 2 | Parallel data loading threads |

### 4. Test Model
Evaluates your trained model on new data.

```bash
# Via menu: Option 4
```

**What it does:**
- Loads your trained `.pt` model
- Tests on images, folders, videos, or webcam
- Adjustable confidence threshold
- Shows detection results with confidence scores

**Testing Options:**
1. Single image file
2. Folder of images
3. Video file
4. Webcam (real-time)
5. Image URL

### 5. Complete Pipeline
Runs Augment → Split → Train in sequence.

```bash
# Via menu: Option 5
```

Perfect for starting from scratch with raw images.

### 6. Quick Train
Skips augmentation and splitting, trains immediately.

```bash
# Via menu: Option 6
```

Use this when your dataset is already organized with train/val/test splits.

---

## ⚙️ Configuration

### `config.py` - Main Settings

```python
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
```

### Supported Models

| Model | Description | Best For |
|-------|-------------|----------|
| `yolo26s.pt` | YOLO26 small | **Recommended** - NMS-free, 43% faster CPU |
| `yolo26n.pt` | YOLO26 nano | Edge devices, mobile |
| `yolov10n.pt` | YOLOv10 nano | Fast inference |
| `yolov8s.pt` | YOLOv8 small | Balanced performance |
| `yolov9t.pt` | YOLOv9 tiny | Ultra-lightweight |

---

## 🛠️ Troubleshooting

### "No module named 'executorch'"
- **Issue**: ExecuTorch not installed
- **Fix**: Use TFLite export instead or skip export

### "CUDA error: no kernel image available"
- **Issue**: PyTorch version doesn't support RTX 5050
- **Fix**: Install PyTorch with CUDA 12.8 support
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### "Dataset 'val:' key missing"
- **Issue**: Your `data.yaml` is incomplete
- **Fix**: Let the pipeline auto-create it via Option 2

### "No labels found in detect set"
- **Issue**: Labels missing or wrong folder structure
- **Fix**: Use Option 2 to reorganize your dataset

### Model download fails
- **Issue**: Network or SSL issue
- **Fix**: Manually download from GitHub releases:
```
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt
```
Place the file in your project folder.

---

## 📝 Examples

### Example 1: Training from Scratch

```bash
# 1. Place your images in a folder with labels
my_dataset/
├── images/
│   ├── fire1.jpg
│   ├── fire2.jpg
│   └── ...
└── labels/
    ├── fire1.txt
    ├── fire2.txt
    └── ...

# 2. Run the pipeline
python main.py
# Select option 5 (Complete Pipeline)

# 3. Follow prompts:
#    - Enter input folder: my_dataset
#    - Enter output folder: (press Enter for default)
#    - Set augmentation target: 2000
#    - Select model: yolo26s.pt
#    - Set epochs: 150

# 4. Wait for training to complete
# 5. Test your model with option 4
```

### Example 2: Quick Test After Training

```bash
python main.py
# Select option 4 (Test Model)

# Enter model path:
# C:\Users\...\runs\detect\YOLO_Training\experiment_1\weights\best.pt

# Set confidence threshold: 0.7

# Choose test option: 1 (Single image)
# Enter image path: test_flame.jpg

# See results:
# ✅ Detected 2 objects:
#    1. Class: 0, Confidence: 85.3%
#    2. Class: 0, Confidence: 72.1%
```

### Example 3: Command Line Mode

```bash
# Quick train
python main.py --mode quick

# Train with specific parameters
python main.py --mode train --dataset my_dataset --model yolo26s.pt --epochs 100 --batch 8

# Test with custom confidence
python main.py --mode test --conf 0.7
```

---

## 📊 Output Files

After training, your results are saved to:

```
runs/detect/
└── YOLO_Training/
    └── experiment_1/
        ├── weights/
        │   ├── best.pt      # Best model (use this for testing)
        │   └── last.pt      # Final epoch model
        ├── results.png      # Training graphs
        ├── results.csv      # Raw training metrics
        └── args.yaml        # Training arguments
```

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics) for YOLO implementation
- [PyTorch](https://pytorch.org) for deep learning framework
- OpenCV for image processing

---

## 📄 License

This project is open-source and available under the MIT License.

---

**Happy Training! 🎉**