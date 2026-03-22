#!/usr/bin/env python3
"""
Dataset splitting for YOLO format (80% train, 10% val, 10% test)
"""

import random
import shutil
from pathlib import Path
import sys
import yaml
from config import SPLIT_RATIOS, print_banner
from utils import get_folder_path, get_output_path, update_data_yaml

def main():
    print_banner("Dataset Splitting (80% Train, 10% Val, 10% Test)")
    
    # Get input folder
    print("\n📁 Your folder should have 'images/' and 'labels/' with ALL files")
    input_path = get_folder_path("Enter input folder path")
    
    # Check structure
    images_path = input_path / 'images'
    labels_path = input_path / 'labels'
    
    if not images_path.exists():
        print(f"❌ 'images' folder not found in {input_path}")
        sys.exit(1)
    if not labels_path.exists():
        print(f"❌ 'labels' folder not found in {input_path}")
        sys.exit(1)
    
    # Get output folder
    default_output = get_output_path(input_path, "split")
    print(f"\n💡 Suggested output: {default_output}")
    output_path = get_folder_path("Enter output folder path (or press Enter for suggested)")
    
    train_ratio, val_ratio, test_ratio = SPLIT_RATIOS
    
    print(f"\n📊 Split ratios:")
    print(f"   Train: {train_ratio:.1%}")
    print(f"   Val:   {val_ratio:.1%}")
    print(f"   Test:  {test_ratio:.1%}")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Find all images
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        images.extend(list(images_path.glob(ext)))
    
    print(f"\n📸 Found {len(images)} images")
    
    if len(images) == 0:
        print("❌ No images found!")
        sys.exit(1)
    
    # Filter images with labels
    valid_images = []
    for img in images:
        lbl = labels_path / f"{img.stem}.txt"
        if lbl.exists():
            valid_images.append(img)
    
    print(f"   Images with labels: {len(valid_images)}")
    
    if len(valid_images) == 0:
        print("❌ No images with matching labels found!")
        sys.exit(1)
    
    # Shuffle and split
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
    
    # Copy files
    for split_name, images_list in [('train', train_images), ('val', val_images), ('test', test_images)]:
        for img in images_list:
            shutil.copy(img, output_path / 'images' / split_name / img.name)
            lbl = labels_path / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy(lbl, output_path / 'labels' / split_name / lbl.name)
    
    # Get class names from labels
    class_ids = set()
    class_names_dict = {}
    
    for img in valid_images:
        lbl = labels_path / f"{img.stem}.txt"
        with open(lbl, 'r') as f:
            for line in f:
                if line.strip():
                    class_ids.add(int(line.split()[0]))
    
    # Create class names
    class_names = [f"class_{i}" for i in sorted(class_ids)]
    
    # Create data.yaml
    yaml_path = update_data_yaml(output_path, class_names)
    
    print(f"\n✅ Dataset split complete!")
    print(f"   Data config: {yaml_path}")
    print(f"   Classes: {len(class_names)}")
    
    # Show structure
    print(f"\n📁 Output structure:")
    for split in ['train', 'val', 'test']:
        img_count = len(list((output_path / 'images' / split).glob('*')))
        lbl_count = len(list((output_path / 'labels' / split).glob('*.txt')))
        print(f"   images/{split}: {img_count} files")
        print(f"   labels/{split}: {lbl_count} files")

if __name__ == '__main__':
    main()