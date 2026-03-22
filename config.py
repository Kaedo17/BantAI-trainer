#!/usr/bin/env python3
"""
Shared configuration for YOLO pipeline
"""

import torch
from pathlib import Path

# Model settings
DEFAULT_MODEL = "yolov26s.pt"  # Use YOLOv26s
AVAILABLE_MODELS = [
    "yolov26n.pt",
    "yolov26s.pt", 
    "yolov26m.pt",
    "yolov26l.pt",
    "yolov26x.pt"
]

# Training defaults
DEFAULT_EPOCHS = 150
DEFAULT_BATCH = 16
DEFAULT_IMGSZ = 640
DEFAULT_WORKERS = 2
DEFAULT_PATIENCE = 50

# Augmentation defaults
AUGMENTATIONS = [
    "flipH", "flipV", "dark", "bright", "gray", "blur", "contrast"
]
DEFAULT_AUG_TARGET = 2000

# Split ratios (train/val/test)
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # 80% train, 10% val, 10% test

def get_gpu_info():
    """Get GPU information if available"""
    if torch.cuda.is_available():
        return {
            'name': torch.cuda.get_device_name(0),
            'vram_gb': torch.cuda.get_device_properties(0).total_memory / 1e9
        }
    return None

def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)