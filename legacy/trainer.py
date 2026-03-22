#!/usr/bin/env python3
"""
YOLO Training Pipeline - Complete workflow for augmentation, splitting, and training
Supports YOLOv26s model with standard YOLO folder structure:
    sourcefolder/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
"""

import os
import cv2
import yaml
import shutil
import random
import argparse
import subprocess
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import torch

class YOLOTrainingPipeline:
    def __init__(self):
        self.working_dir = None
        self.dataset_path = None
        self.output_path = None
        self.model = None
        
    def print_banner(self, text):
        """Print a formatted banner"""
        print("\n" + "="*60)
        print(f"  {text}")
        print("="*60)
    
    def get_folder_path(self, prompt):
        """Get folder path from user with validation"""
        while True:
            path = input(f"{prompt}: ").strip().strip('"')
            if not path:
                print("❌ Path cannot be empty. Please try again.")
                continue
            path = Path(path)
            if not path.exists():
                print(f"❌ Path '{path}' does not exist. Please try again.")
                continue
            if not path.is_dir():
                print(f"❌ '{path}' is not a directory. Please try again.")
                continue
            return path
    
    def validate_yolo_structure(self, folder_path):
        """Check if folder has valid YOLO structure with train/val/test splits"""
        images_path = folder_path / 'images'
        labels_path = folder_path / 'labels'
        
        if not images_path.exists():
            print(f"❌ 'images' folder not found in {folder_path}")
            return False, None, None
            
        if not labels_path.exists():
            print(f"❌ 'labels' folder not found in {folder_path}")
            return False, None, None
        
        # Check for train/val/test subfolders
        splits = ['train', 'val', 'test']
        found_splits = []
        
        for split in splits:
            if (images_path / split).exists() and (labels_path / split).exists():
                found_splits.append(split)
        
        if not found_splits:
            print(f"⚠️ No train/val/test subfolders found in {folder_path}")
            print("   Expected structure:")
            print("   sourcefolder/images/train, sourcefolder/images/val, sourcefolder/images/test")
            print("   sourcefolder/labels/train, sourcefolder/labels/val, sourcefolder/labels/test")
            return False, None, None
        
        print(f"   Found splits: {', '.join(found_splits)}")
        
        # Count images
        for split in found_splits:
            img_count = len(list((images_path / split).glob('*.*')))
            lbl_count = len(list((labels_path / split).glob('*.txt')))
            print(f"   {split}: {img_count} images, {lbl_count} labels")
        
        return True, images_path, labels_path
    
    def augment_images(self):
        """Augment images with various transformations"""
        self.print_banner("Image Augmentation")
        
        # Get input folder
        print("\n📁 Augmentation Configuration")
        print("   Your folder should have 'images/train' and 'labels/train' subdirectories")
        aug_input = self.get_folder_path("Enter input folder path")
        
        # Validate input structure
        valid, images_path, labels_path = self.validate_yolo_structure(aug_input)
        if not valid:
            print("❌ Invalid folder structure. Input must have 'images/train' and 'labels/train' folders.")
            return False
        
        # Get output folder
        aug_output = self.get_folder_path("Enter output folder path (where augmented data will be saved)")
        
        # Get augmentation settings
        try:
            target_limit = int(input("Enter target number of augmented images (default: 2000): ").strip() or "2000")
        except ValueError:
            target_limit = 2000
        
        print(f"\n🚀 Starting augmentation...")
        print(f"   Input: {aug_input}")
        print(f"   Output: {aug_output}")
        print(f"   Target: {target_limit} images")
        
        # Create output directories with same structure
        out_img_dir = aug_output / 'images' / 'train'
        out_lbl_dir = aug_output / 'labels' / 'train'
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Also copy val/test if they exist
        for split in ['val', 'test']:
            src_img = images_path / split
            src_lbl = labels_path / split
            if src_img.exists() and src_lbl.exists():
                dst_img = aug_output / 'images' / split
                dst_lbl = aug_output / 'labels' / split
                dst_img.mkdir(parents=True, exist_ok=True)
                dst_lbl.mkdir(parents=True, exist_ok=True)
                # Copy all files
                for img in src_img.glob('*.*'):
                    shutil.copy(img, dst_img / img.name)
                for lbl in src_lbl.glob('*.txt'):
                    shutil.copy(lbl, dst_lbl / lbl.name)
                print(f"   Copied existing {split} split")
        
        # Define augmentation functions
        def flip_labels_horizontal(lines):
            new = []
            for line in lines:
                parts = line.split()
                cls, x, y, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x = 1.0 - x
                new.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            return new
        
        def flip_labels_vertical(lines):
            new = []
            for line in lines:
                parts = line.split()
                cls, x, y, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                y = 1.0 - y
                new.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            return new
        
        # Find all training images
        train_images_path = images_path / 'train'
        train_labels_path = labels_path / 'train'
        
        if not train_images_path.exists():
            print("❌ No 'train' folder found in images directory!")
            return False
        
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            images.extend(list(train_images_path.glob(ext)))
        
        print(f"\n📸 Found {len(images)} source training images")
        
        if len(images) == 0:
            print("❌ No images found in training folder!")
            return False
        
        total_saved = 0
        for img_path in images:
            if total_saved >= target_limit:
                break
            
            stem = img_path.stem
            lbl_path = train_labels_path / f"{stem}.txt"
            
            if not lbl_path.exists():
                continue
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            with open(lbl_path, 'r') as f:
                label_lines = [l.strip() for l in f.readlines() if l.strip()]
            
            if not label_lines:
                continue
            
            # Define augmentations
            augs = [
                ("orig", img, label_lines),
                ("flipH", cv2.flip(img, 1), flip_labels_horizontal(label_lines)),
                ("flipV", cv2.flip(img, 0), flip_labels_vertical(label_lines)),
                ("dark", cv2.convertScaleAbs(img, alpha=0.4, beta=0), label_lines),
                ("bright", cv2.convertScaleAbs(img, alpha=1.6, beta=30), label_lines),
                ("gray", cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), label_lines),
                ("blur", cv2.GaussianBlur(img, (5, 5), 0), label_lines),
                ("contrast", cv2.convertScaleAbs(img, alpha=1.8, beta=-30), label_lines)
            ]
            
            for suffix, augmented_img, augmented_labels in augs:
                if total_saved < target_limit:
                    # Save image
                    out_path = out_img_dir / f"{stem}_{suffix}.jpg"
                    cv2.imwrite(str(out_path), augmented_img)
                    
                    # Save labels
                    out_lbl_path = out_lbl_dir / f"{stem}_{suffix}.txt"
                    with open(out_lbl_path, 'w') as f:
                        f.write("\n".join(augmented_labels) + "\n")
                    
                    total_saved += 1
                    
                    if total_saved % 100 == 0:
                        print(f"   Progress: {total_saved}/{target_limit} images")
        
        print(f"\n✅ Augmentation complete! Saved {total_saved} images")
        print(f"   Images: {out_img_dir}")
        print(f"   Labels: {out_lbl_dir}")
        
        # Update data.yaml
        data_yaml_path = aug_output / 'data.yaml'
        if not data_yaml_path.exists():
            # Create data.yaml
            data_yaml = {
                'path': str(aug_output),
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'nc': 1,  # Will be updated later
                'names': ['Open-Flame-Hazard']
            }
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data_yaml, f, default_flow_style=False)
            print(f"   Created data.yaml at {data_yaml_path}")
        
        return True
    
    def split_dataset(self):
        """Split dataset into train/val/test sets (80/10/10) - Use if your data isn't already split"""
        self.print_banner("Dataset Splitting (80% Train, 10% Val, 10% Test)")
        
        # Get input folder (should have images/ and labels/ with all files, no split yet)
        print("\n📁 Split Configuration")
        print("   Your folder should have 'images/' and 'labels/' folders with ALL files (no train/val subfolders)")
        split_input = self.get_folder_path("Enter input folder path")
        
        # Check if it's already split
        if (split_input / 'images' / 'train').exists():
            print("⚠️ This folder already has train/val/test splits!")
            print("   If you want to re-split, please use a folder with unsplit data.")
            choice = input("   Continue anyway? (y/n): ").strip().lower()
            if choice != 'y':
                return False
        
        # Get output folder
        split_output = self.get_folder_path("Enter output folder path (where split data will be saved)")
        
        # Fixed split ratios: 80% train, 10% val, 10% test
        train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
        
        print(f"\n📊 Split ratios:")
        print(f"   Train: {train_ratio:.1%}")
        print(f"   Val:   {val_ratio:.1%}")
        print(f"   Test:  {test_ratio:.1%}")
        
        # Create output directories with YOLO structure
        for split in ['train', 'val', 'test']:
            (split_output / 'images' / split).mkdir(parents=True, exist_ok=True)
            (split_output / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Find images - try to find them in images/ or images/train
        images_path = split_input / 'images'
        if (images_path / 'train').exists():
            images_path = images_path / 'train'
        
        # Find all images
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            images.extend(list(images_path.glob(ext)))
        
        print(f"\n📸 Found {len(images)} images")
        
        if len(images) == 0:
            print("❌ No images found in input folder!")
            return False
        
        # Find labels
        labels_path = split_input / 'labels'
        if (labels_path / 'train').exists():
            labels_path = labels_path / 'train'
        
        # Filter images that have corresponding labels
        valid_images = []
        for img in images:
            lbl = labels_path / f"{img.stem}.txt"
            if lbl.exists():
                valid_images.append(img)
        
        print(f"   Images with labels: {len(valid_images)}")
        
        if len(valid_images) == 0:
            print("❌ No images with matching labels found!")
            return False
        
        # Shuffle and split (80/10/10)
        random.shuffle(valid_images)
        n_train = int(len(valid_images) * train_ratio)
        n_val = int(len(valid_images) * val_ratio)
        
        train_images = valid_images[:n_train]
        val_images = valid_images[n_train:n_train + n_val]
        test_images = valid_images[n_train + n_val:]
        
        print(f"\n📊 Split results:")
        print(f"   Train: {len(train_images)}")
        print(f"   Val:   {len(val_images)}")
        print(f"   Test:  {len(test_images)}")
        
        # Copy files to their respective folders
        for split_name, images_list in [('train', train_images), ('val', val_images), ('test', test_images)]:
            for img in images_list:
                # Copy image
                shutil.copy(img, split_output / 'images' / split_name / img.name)
                # Copy label
                lbl = labels_path / f"{img.stem}.txt"
                if lbl.exists():
                    shutil.copy(lbl, split_output / 'labels' / split_name / lbl.name)
        
        # Get class names from labels
        class_ids = set()
        for img in valid_images:
            lbl = labels_path / f"{img.stem}.txt"
            with open(lbl, 'r') as f:
                for line in f:
                    if line.strip():
                        class_ids.add(int(line.split()[0]))
        
        # Create data.yaml
        data_yaml = {
            'path': str(split_output),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_ids),
            'names': [f'class_{i}' for i in sorted(class_ids)]
        }
        
        yaml_path = split_output / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"\n✅ Dataset split complete!")
        print(f"   Data config: {yaml_path}")
        print(f"   Classes: {len(class_ids)}")
        
        # Show folder structure
        print(f"\n📁 Output structure:")
        for split in ['train', 'val', 'test']:
            img_count = len(list((split_output / 'images' / split).glob('*')))
            lbl_count = len(list((split_output / 'labels' / split).glob('*.txt')))
            print(f"   images/{split}: {img_count} files")
            print(f"   labels/{split}: {lbl_count} files")
        
        return True
    
    def train_model(self):
        """Train YOLO model"""
        self.print_banner("Model Training")
        
        # Get dataset path (should have data.yaml)
        dataset_path = self.get_folder_path("Enter path to dataset folder (with data.yaml)")
        data_yaml = dataset_path / 'data.yaml'
        
        if not data_yaml.exists():
            print(f"❌ data.yaml not found in {dataset_path}")
            return False
        
        # Get training parameters
        print("\n⚙️ Training Configuration")
        try:
            epochs = int(input("Enter number of epochs (default: 150): ").strip() or "150")
            batch = int(input("Enter batch size (default: 16): ").strip() or "16")
            imgsz = int(input("Enter image size (default: 640): ").strip() or "640")
            workers = int(input("Enter number of workers (default: 2): ").strip() or "2")
        except ValueError:
            epochs, batch, imgsz, workers = 150, 16, 640, 2
        
        # Get model name
        model_name = input("Enter model name (default: yolov26s.pt): ").strip() or "yolov26s.pt"
        if not model_name.endswith('.pt'):
            model_name += '.pt'
        
        # Get project name
        project_name = input("Enter project name (default: YOLO_Training): ").strip() or "YOLO_Training"
        experiment_name = input("Enter experiment name (default: experiment_1): ").strip() or "experiment_1"
        
        # Check GPU
        if torch.cuda.is_available():
            print(f"\n🎮 GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("\n⚠️ No GPU detected. Training will be slow on CPU.")
        
        print(f"\n🚀 Starting training...")
        print(f"   Model: {model_name}")
        print(f"   Dataset: {data_yaml}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch}")
        print(f"   Image size: {imgsz}")
        
        # Load model
        try:
            model = YOLO(model_name)
        except Exception as e:
            print(f"⚠️ Could not load {model_name}, trying yolov8n.pt...")
            model = YOLO("yolov8n.pt")
        
        # Train model
        try:
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
                patience=50,
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
            
            # Export options
            export_choice = input("\n📦 Export model? (y/n): ").strip().lower()
            if export_choice == 'y':
                print("\n📦 Export formats:")
                print("   1. ONNX")
                print("   2. TorchScript")
                print("   3. TFLite")
                print("   4. All formats")
                
                export_opt = input("Choose export format (1-4): ").strip()
                best_path = results.save_dir / "weights" / "best.pt"
                best_model = YOLO(str(best_path))
                
                if export_opt == '1':
                    best_model.export(format="onnx")
                    print("✅ Exported to ONNX")
                elif export_opt == '2':
                    best_model.export(format="torchscript")
                    print("✅ Exported to TorchScript")
                elif export_opt == '3':
                    best_model.export(format="tflite")
                    print("✅ Exported to TFLite")
                elif export_opt == '4':
                    best_model.export(format="onnx")
                    best_model.export(format="torchscript")
                    best_model.export(format="tflite")
                    print("✅ Exported to all formats")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def run_pipeline(self):
        """Main pipeline runner"""
        self.print_banner("YOLO Training Pipeline")
        
        while True:
            print("\n📋 Available Options:")
            print("   1. Augment Images (from train folder)")
            print("   2. Split Dataset (if your data is not already split)")
            print("   3. Train Model")
            print("   4. Run Complete Pipeline (Augment → Split → Train)")
            print("   5. Exit")
            
            choice = input("\nSelect an option (1-5): ").strip()
            
            if choice == '1':
                self.augment_images()
            elif choice == '2':
                self.split_dataset()
            elif choice == '3':
                self.train_model()
            elif choice == '4':
                print("\n📋 Running complete pipeline...")
                print("   Step 1: Augmentation")
                if self.augment_images():
                    print("\n✅ Augmentation complete! Proceeding to splitting...")
                    print("\n   Step 2: Dataset Splitting")
                    if self.split_dataset():
                        print("\n✅ Splitting complete! Proceeding to training...")
                        print("\n   Step 3: Model Training")
                        self.train_model()
            elif choice == '5':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please try again.")
            
            input("\nPress Enter to continue...")


def main():
    """Main entry point"""
    pipeline = YOLOTrainingPipeline()
    pipeline.run_pipeline()


if __name__ == '__main__':
    main()