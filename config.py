#!/usr/bin/env python3
"""
Shared configuration for YOLO pipeline
"""

import torch
from pathlib import Path

# Model settings - CORRECTED NAMES (yolo26s.pt not yolov26s.pt)
DEFAULT_MODEL = "yolo26s.pt"  # Use YOLO26s (NMS-free model)
AVAILABLE_MODELS = [
    "yolo26n.pt",   # nano
    "yolo26s.pt",   # small - recommended
    "yolo26m.pt",   # medium
    "yolo26l.pt",   # large
    "yolo26x.pt",   # x-large
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

# YOLO Model URLs (Official Ultralytics assets)
YOLO_MODEL_URLS = {
    # YOLO26 series (latest - NMS-free)
    'yolo26n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt',
    'yolo26s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt',
    'yolo26m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt',
    'yolo26l.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt',
    'yolo26x.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt',
    
    # YOLOv8 series
    'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt',
    'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt',
    'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt',
    'yolov8l.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8l.pt',
    'yolov8x.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x.pt',
    
    # YOLOv9 series
    'yolov9t.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9t.pt',
    'yolov9s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9s.pt',
    'yolov9m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9m.pt',
    'yolov9c.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9c.pt',
    'yolov9e.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9e.pt',
    
    # YOLOv10 series
    'yolov10n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10n.pt',
    'yolov10s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10s.pt',
    'yolov10m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10m.pt',
    'yolov10b.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10b.pt',
    'yolov10l.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10l.pt',
    'yolov10x.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10x.pt',
}

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

def download_model_if_needed(model_name: str, save_path: Path = None) -> bool:
    """Download model if it doesn't exist locally"""
    import urllib.request
    import ssl
    
    if save_path is None:
        save_path = Path.cwd()
    
    model_path = save_path / model_name
    
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        return True
    
    if model_name not in YOLO_MODEL_URLS:
        print(f"⚠️ Unknown model: {model_name}")
        print(f"   Available models: {', '.join(list(YOLO_MODEL_URLS.keys()))}")
        return False
    
    url = YOLO_MODEL_URLS[model_name]
    print(f"📥 Downloading {model_name}...")
    print(f"   From: {url}")
    print(f"   To: {model_path}")
    
    try:
        # Disable SSL verification (sometimes needed)
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            bar_length = 40
            filled_length = int(bar_length * downloaded / total_size)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"\r   [{bar}] {percent:.1f}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, model_path, report_progress)
        print()  # New line after progress bar
        
        if model_path.exists() and model_path.stat().st_size > 0:
            print(f"✅ Successfully downloaded {model_name}")
            return True
        else:
            print(f"❌ Download failed - file is empty")
            return False
            
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print(f"   You can manually download from: {url}")
        return False