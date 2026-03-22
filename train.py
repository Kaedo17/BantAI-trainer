#!/usr/bin/env python3
"""
YOLO model training with automatic structure detection
"""

import sys
import torch
from pathlib import Path
from ultralytics import YOLO
from config import (
    DEFAULT_MODEL, DEFAULT_EPOCHS, DEFAULT_BATCH, 
    DEFAULT_IMGSZ, DEFAULT_WORKERS, DEFAULT_PATIENCE,
    get_gpu_info, print_banner
)
from utils import get_folder_path, detect_folder_structure, validate_and_show_structure

def main():
    print_banner("Model Training")
    
    # Get dataset path
    dataset_path = get_folder_path("Enter path to dataset folder")
    data_yaml = dataset_path / 'data.yaml'
    
    # If data.yaml doesn't exist, check if we can create it
    if not data_yaml.exists():
        print(f"\n⚠️  data.yaml not found in {dataset_path}")
        print("   Checking folder structure...")
        
        structure = detect_folder_structure(dataset_path)
        
        if structure['has_train_val_test']:
            print("   ✅ Found valid YOLO structure with train/val/test splits")
            print("   You can still train, but you need a data.yaml file.")
            
            # Create data.yaml on the fly
            from utils import update_data_yaml
            from config import SPLIT_RATIOS
            
            # Try to get class names from labels
            class_names = []
            if structure['type'] == 'split':
                labels_path = structure['labels_path'] / 'train'
                if labels_path.exists():
                    # Extract class IDs from labels
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
            sys.exit(1)
    
    # Load and display dataset info
    import yaml
    with open(data_yaml, 'r') as f:
        dataset_info = yaml.safe_load(f)
    
    print(f"\n📊 Dataset info:")
    print(f"   Classes: {dataset_info.get('nc', 'unknown')}")
    print(f"   Class names: {dataset_info.get('names', 'unknown')}")
    
    # Get model selection
    print(f"\n📦 Available models:")
    models = [
        "yolov26n.pt (nano)",
        "yolov26s.pt (small - recommended)", 
        "yolov26m.pt (medium)",
        "yolov26l.pt (large)",
        "yolov26x.pt (x-large)"
    ]
    for m in models:
        print(f"   {m}")
    
    model_choice = input(f"\nEnter model name (default: {DEFAULT_MODEL}): ").strip()
    if not model_choice:
        model_choice = DEFAULT_MODEL
    if not model_choice.endswith('.pt'):
        model_choice += '.pt'
    
    # Get training parameters
    print("\n⚙️ Training Configuration")
    
    try:
        epochs = int(input(f"Enter number of epochs (default: {DEFAULT_EPOCHS}): ").strip() or str(DEFAULT_EPOCHS))
        batch = int(input(f"Enter batch size (default: {DEFAULT_BATCH}): ").strip() or str(DEFAULT_BATCH))
        imgsz = int(input(f"Enter image size (default: {DEFAULT_IMGSZ}): ").strip() or str(DEFAULT_IMGSZ))
        workers = int(input(f"Enter number of workers (default: {DEFAULT_WORKERS}): ").strip() or str(DEFAULT_WORKERS))
    except ValueError:
        epochs, batch, imgsz, workers = DEFAULT_EPOCHS, DEFAULT_BATCH, DEFAULT_IMGSZ, DEFAULT_WORKERS
    
    # Get project name
    project_name = input("Enter project name (default: YOLO_Training): ").strip() or "YOLO_Training"
    experiment_name = input("Enter experiment name (default: experiment_1): ").strip() or "experiment_1"
    
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
            project=project_name,
            name=experiment_name,
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
        print(f"   Model saved to: {results.save_dir}")
        
        # Show model info
        best_path = results.save_dir / "weights" / "best.pt"
        print(f"   Best model: {best_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()