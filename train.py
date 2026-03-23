#!/usr/bin/env python3
"""
YOLO model training with automatic structure detection and model download
Output saved to Trained/ModelName folder
"""

import sys
import torch
from pathlib import Path
from ultralytics import YOLO
import yaml

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_MODEL, DEFAULT_EPOCHS, DEFAULT_BATCH, 
    DEFAULT_IMGSZ, DEFAULT_WORKERS, DEFAULT_PATIENCE,
    get_gpu_info, print_banner, download_model_if_needed, YOLO_MODEL_URLS
)
from utils import get_folder_path, detect_folder_structure, update_data_yaml, ROOT_DIR

# Define training output base
TRAINED_BASE = ROOT_DIR / "Trained"

def ensure_trained_base():
    """Ensure the Trained directory exists"""
    TRAINED_BASE.mkdir(parents=True, exist_ok=True)
    return TRAINED_BASE

def get_available_models():
    """Get list of available YOLO models grouped by version"""
    models_by_version = {}
    
    for model_name in sorted(YOLO_MODEL_URLS.keys()):
        if model_name.startswith('yolo26'):
            version = 'YOLO26 (NMS-free, Edge-optimized)'
            size_code = model_name.replace('yolo26', '').replace('.pt', '')
        elif model_name.startswith('yolov10'):
            version = 'YOLOv10'
            size_code = model_name.replace('yolov10', '').replace('.pt', '')
        elif model_name.startswith('yolov9'):
            version = 'YOLOv9'
            size_code = model_name.replace('yolov9', '').replace('.pt', '')
        elif model_name.startswith('yolov8'):
            version = 'YOLOv8'
            size_code = model_name.replace('yolov8', '').replace('.pt', '')
        else:
            continue
        
        size_names = {
            'n': 'nano', 's': 'small', 'm': 'medium', 'l': 'large', 'x': 'x-large',
            't': 'tiny', 'c': 'compact', 'b': 'balanced', 'e': 'extra'
        }
        size_name = size_names.get(size_code, size_code)
        
        if version not in models_by_version:
            models_by_version[version] = []
        models_by_version[version].append((model_name, size_name))
    
    return models_by_version

def main():
    print_banner("Model Training")
    
    # Get dataset path with navigation
    while True:
        dataset_path = get_folder_path("Enter path to dataset folder", allow_cancel=True)
        if dataset_path == 'BACK':
            return
        if dataset_path == 'CANCEL':
            print("❌ Cancelled by user")
            return
        if dataset_path:
            break
    
    data_yaml = dataset_path / 'data.yaml'
    
    # If data.yaml doesn't exist, check if we can create it
    if not data_yaml.exists():
        print(f"\n⚠️  data.yaml not found in {dataset_path}")
        print("   Checking folder structure...")
        
        structure = detect_folder_structure(dataset_path)
        
        if structure['has_train_val_test']:
            print("   ✅ Found valid YOLO structure with train/val/test splits")
            
            # Create data.yaml on the fly
            class_names = []
            if structure['type'] == 'split':
                labels_path = structure['labels_path'] / 'train'
                if labels_path.exists():
                    class_ids = set()
                    for lbl_file in labels_path.glob('*.txt'):
                        with open(lbl_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    try:
                                        class_ids.add(int(line.split()[0]))
                                    except:
                                        pass
                    class_names = [f"class_{i}" for i in sorted(class_ids)]
            
            if not class_names:
                class_names = ['class_0']
            
            data_yaml = update_data_yaml(dataset_path, class_names)
            print(f"   Created data.yaml at {data_yaml}")
        else:
            print("❌ Could not find valid YOLO structure")
            print("   Expected: images/train, images/val, labels/train, labels/val")
            return  # FIXED: replaced 'continue' with 'return'
    
    # Load and display dataset info
    with open(data_yaml, 'r') as f:
        dataset_info = yaml.safe_load(f)
    
    print(f"\n📊 Dataset info:")
    print(f"   Classes: {dataset_info.get('nc', 'unknown')}")
    print(f"   Class names: {dataset_info.get('names', 'unknown')}")
    
    # Get model selection
    print(f"\n📦 Available YOLO models:")
    
    models_by_version = get_available_models()
    for version, models in models_by_version.items():
        print(f"\n   {version}:")
        for model_name, size_name in models:
            default_marker = " (recommended)" if model_name == DEFAULT_MODEL else ""
            print(f"      {model_name} - {size_name}{default_marker}")
    
    model_choice = input(f"\nEnter model name (default: {DEFAULT_MODEL}): ").strip()
    if not model_choice:
        model_choice = DEFAULT_MODEL
    if not model_choice.endswith('.pt'):
        model_choice += '.pt'
    
    # Download model if not available
    if not download_model_if_needed(model_choice):
        print("\n❌ Failed to get model. Please check the model name and try again.")
        return
    
    # Get training parameters
    print("\n⚙️ Training Configuration")
    
    try:
        epochs = int(input(f"Enter number of epochs (default: {DEFAULT_EPOCHS}): ").strip() or str(DEFAULT_EPOCHS))
        batch = int(input(f"Enter batch size (default: {DEFAULT_BATCH}): ").strip() or str(DEFAULT_BATCH))
        imgsz = int(input(f"Enter image size (default: {DEFAULT_IMGSZ}): ").strip() or str(DEFAULT_IMGSZ))
        workers = int(input(f"Enter number of workers (default: {DEFAULT_WORKERS}): ").strip() or str(DEFAULT_WORKERS))
    except ValueError:
        epochs, batch, imgsz, workers = DEFAULT_EPOCHS, DEFAULT_BATCH, DEFAULT_IMGSZ, DEFAULT_WORKERS
    
    # Get output folder name (will be created in Trained directory)
    ensure_trained_base()
    print(f"\n📁 Output will be saved in: {TRAINED_BASE}")
    model_name_input = input("Enter model name for output folder (e.g., Fire_Detector_v1): ").strip()
    if not model_name_input:
        model_name_input = "YOLO_Model"
    
    output_path = TRAINED_BASE / model_name_input
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Models will be saved to: {output_path}")
    
    # GPU info
    gpu = get_gpu_info()
    if gpu:
        print(f"\n🎮 GPU detected: {gpu['name']}")
        print(f"   VRAM: {gpu['vram_gb']:.2f} GB")
    else:
        print("\n⚠️ No GPU detected. Training will be slow on CPU.")
    
    print(f"\n🚀 Starting training...")
    print(f"   Model: {model_choice}")
    print(f"   Dataset: {data_yaml}")
    print(f"   Output: {output_path}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch}")
    print(f"   Image size: {imgsz}")
    
    # Load and train model
    try:
        print(f"\n📥 Loading model {model_choice}...")
        model = YOLO(model_choice)
        
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            augment=True,
            workers=workers,
            batch=batch,
            device=0 if torch.cuda.is_available() else 'cpu',
            project=str(output_path),
            name="weights",
            patience=DEFAULT_PATIENCE,
            save=True,
            exist_ok=True,
            optimizer='auto',
            verbose=True,
            amp=True,
            cache=False,
            close_mosaic=10,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0
        )
        
        print(f"\n✅ Training complete!")
        print(f"   Model saved to: {output_path}")
        
        # Show model info
        best_path = output_path / "weights" / "best.pt"
        if best_path.exists():
            print(f"   Best model: {best_path}")
        else:
            alt_path = output_path / "weights" / "weights" / "best.pt"
            if alt_path.exists():
                print(f"   Best model: {alt_path}")
            else:
                print(f"   Check in: {output_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

if __name__ == '__main__':
    main()