#!/usr/bin/env python3
"""
Dataset splitting for YOLO format with automatic structure detection
Output saved to Output/Split/ folder with preserved class information
"""

import random
import shutil
import yaml
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import SPLIT_RATIOS, print_banner
from utils import (
    get_folder_path, detect_folder_structure,
    validate_and_show_structure, ROOT_DIR, OUTPUT_BASE,
    copy_data_yaml, update_data_yaml
)

# Define split base directory
SPLIT_BASE = OUTPUT_BASE / "Split"

def ensure_split_base():
    """Ensure the Split directory exists"""
    SPLIT_BASE.mkdir(parents=True, exist_ok=True)
    return SPLIT_BASE

def normalize_to_yolo_format(input_path: Path, output_path: Path, split_ratios: tuple):
    """
    Convert any detected structure to standard YOLO format, preserving class info
    """
    structure = detect_folder_structure(input_path)
    
    if structure['type'] == 'unknown':
        print("❌ Could not detect folder structure")
        return False, None
    
    # Try to copy existing data.yaml to preserve class info
    source_yaml = input_path / 'data.yaml'
    preserved_config = None
    
    if source_yaml.exists():
        try:
            with open(source_yaml, 'r') as f:
                preserved_config = yaml.safe_load(f)
            print(f"📋 Found existing data.yaml with {preserved_config.get('nc', 0)} classes")
        except Exception as e:
            print(f"⚠️ Could not read data.yaml: {e}")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Collect all images based on structure
    all_images = []
    
    if structure['type'] == 'split':
        # Look for images in train folder (or any available split)
        for split in structure['splits']:
            img_path = structure['images_path'] / split
            if img_path.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    all_images.extend(list(img_path.glob(ext)))
    
    elif structure['type'] == 'split_alt':
        for split in structure['splits']:
            img_path = structure['images_path'] / split / 'images'
            if img_path.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    all_images.extend(list(img_path.glob(ext)))
    
    elif structure['type'] in ['flat', 'mixed']:
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            all_images.extend(list(structure['images_path'].glob(ext)))
    
    if not all_images:
        print("❌ No images found")
        return False, None
    
    # Filter images that have labels
    valid_images = []
    labels_found = set()
    
    for img in all_images:
        # Determine label path based on structure
        if structure['type'] == 'split':
            # For split structure, look in corresponding split folder
            for split in structure['splits']:
                lbl_path = structure['labels_path'] / split / f"{img.stem}.txt"
                if lbl_path.exists():
                    valid_images.append(img)
                    labels_found.add(lbl_path)
                    break
        elif structure['type'] == 'split_alt':
            for split in structure['splits']:
                lbl_path = structure['labels_path'] / split / 'labels' / f"{img.stem}.txt"
                if lbl_path.exists():
                    valid_images.append(img)
                    labels_found.add(lbl_path)
                    break
        else:
            lbl_path = structure['labels_path'] / f"{img.stem}.txt"
            if lbl_path.exists():
                valid_images.append(img)
                labels_found.add(lbl_path)
    
    print(f"   Found {len(valid_images)} images with labels")
    
    if len(valid_images) == 0:
        print("❌ No images with matching labels found")
        print("   Check that your label files (.txt) have the same name as images")
        return False, None
    
    # Shuffle and split
    random.shuffle(valid_images)
    train_ratio, val_ratio, test_ratio = split_ratios
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
        if not images_list:
            print(f"   ⚠️ No images for {split_name} split")
            continue
            
        for img in images_list:
            # Copy image
            shutil.copy(img, output_path / 'images' / split_name / img.name)
            
            # Find and copy label
            if structure['type'] == 'split':
                for split in structure['splits']:
                    lbl = structure['labels_path'] / split / f"{img.stem}.txt"
                    if lbl.exists():
                        shutil.copy(lbl, output_path / 'labels' / split_name / lbl.name)
                        break
            elif structure['type'] == 'split_alt':
                for split in structure['splits']:
                    lbl = structure['labels_path'] / split / 'labels' / f"{img.stem}.txt"
                    if lbl.exists():
                        shutil.copy(lbl, output_path / 'labels' / split_name / lbl.name)
                        break
            else:
                lbl = structure['labels_path'] / f"{img.stem}.txt"
                if lbl.exists():
                    shutil.copy(lbl, output_path / 'labels' / split_name / lbl.name)
    
    # Create data.yaml using preserved config if available
    if preserved_config:
        data_yaml = {
            'path': str(output_path),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': preserved_config.get('nc', 1),
            'names': preserved_config.get('names', ['class_0'])
        }
        print(f"✅ Preserved {preserved_config.get('nc', 0)} classes from source")
    else:
        # Get class names from labels
        class_names = [f"class_{i}" for i in range(1)]  # Default
        data_yaml = {
            'path': str(output_path),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,
            'names': ['class_0']
        }
        print(f"📝 Created default data.yaml")
    
    yaml_path = update_data_yaml(output_path, data_yaml)
    
    return True, yaml_path

def main():
    print_banner("Dataset Splitting (80% Train, 10% Val, 10% Test)")
    
    # Get input folder with navigation
    while True:
        input_path = get_folder_path("Enter input folder path", allow_cancel=True)
        if input_path == 'BACK':
            return
        if input_path == 'CANCEL':
            print("❌ Cancelled by user")
            return
        if input_path:
            break
    
    # Detect and show structure
    print("\n🔍 Analyzing folder structure...")
    structure = validate_and_show_structure(input_path)
    
    if structure['type'] == 'unknown':
        print("❌ Could not detect folder structure")
        print("   Expected structures:")
        print("   1. folder/images/train, folder/images/val, folder/labels/train, folder/labels/val")
        print("   2. folder/train/images, folder/val/images, folder/train/labels, folder/val/labels")
        print("   3. folder/images, folder/labels (flat structure)")
        return
    
    # Get output folder (will be created in Split directory)
    ensure_split_base()
    print(f"\n📁 Output will be saved in: {SPLIT_BASE}")
    
    default_name = f"{input_path.name}_split"
    print(f"   Default name: {default_name}")
    
    folder_name = input("Enter name for split dataset folder: ").strip()
    if not folder_name:
        folder_name = default_name
    
    output_path = SPLIT_BASE / folder_name
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output folder: {output_path}")
    
    # Use the normalize function to handle all cases
    success, yaml_path = normalize_to_yolo_format(input_path, output_path, SPLIT_RATIOS)
    
    if success:
        print(f"\n✅ Dataset split complete!")
        print(f"   Data config: {yaml_path}")
        
        # Show final structure
        print(f"\n📁 Output structure:")
        for split in ['train', 'val', 'test']:
            img_count = len(list((output_path / 'images' / split).glob('*')))
            lbl_count = len(list((output_path / 'labels' / split).glob('*.txt')))
            if img_count > 0 or lbl_count > 0:
                print(f"   images/{split}: {img_count} files")
                print(f"   labels/{split}: {lbl_count} files")
    else:
        print("❌ Failed to split dataset")

if __name__ == '__main__':
    main()