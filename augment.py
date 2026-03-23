#!/usr/bin/env python3
"""
Image augmentation for YOLO datasets with automatic structure detection
Output saved to Output/Augmented/ folder with preserved class information
"""

import cv2
import numpy as np
import shutil
import yaml
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import DEFAULT_AUG_TARGET, print_banner
from utils import (
    get_folder_path, detect_folder_structure, 
    validate_and_show_structure, ROOT_DIR, OUTPUT_BASE,
    copy_data_yaml, create_default_data_yaml
)

# Define augmentation base directory
AUGMENTED_BASE = OUTPUT_BASE / "Augmented"

def ensure_augmented_base():
    """Ensure the Augmented directory exists"""
    AUGMENTED_BASE.mkdir(parents=True, exist_ok=True)
    return AUGMENTED_BASE

def flip_labels_horizontal(lines):
    """Flip YOLO labels horizontally"""
    new = []
    for line in lines:
        parts = line.split()
        cls, x, y, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x = 1.0 - x
        new.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return new

def flip_labels_vertical(lines):
    """Flip YOLO labels vertically"""
    new = []
    for line in lines:
        parts = line.split()
        cls, x, y, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        y = 1.0 - y
        new.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return new

def augment_image(img, labels, aug_type):
    """Apply specific augmentation to image and labels"""
    if aug_type == "flipH":
        return cv2.flip(img, 1), flip_labels_horizontal(labels)
    elif aug_type == "flipV":
        return cv2.flip(img, 0), flip_labels_vertical(labels)
    elif aug_type == "dark":
        return cv2.convertScaleAbs(img, alpha=0.4, beta=0), labels
    elif aug_type == "bright":
        return cv2.convertScaleAbs(img, alpha=1.6, beta=30), labels
    elif aug_type == "gray":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), labels
    elif aug_type == "blur":
        return cv2.GaussianBlur(img, (5, 5), 0), labels
    elif aug_type == "contrast":
        return cv2.convertScaleAbs(img, alpha=1.8, beta=-30), labels
    else:
        return img, labels

def get_train_images(structure, images_path, labels_path):
    """Get training images based on detected structure"""
    images = []
    
    if structure['type'] == 'split':
        train_img_path = images_path / 'train'
        if train_img_path.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                images.extend(list(train_img_path.glob(ext)))
                
    elif structure['type'] == 'split_alt':
        train_img_path = images_path / 'train' / 'images'
        if train_img_path.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                images.extend(list(train_img_path.glob(ext)))
                
    elif structure['type'] in ['flat', 'mixed']:
        print("\n⚠️  Flat structure detected. All images will be used for augmentation.")
        print("   Consider splitting your dataset first if you need separate validation.")
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            images.extend(list(images_path.glob(ext)))
    
    return images

def copy_existing_splits(output_path, images_path, labels_path, structure):
    """Copy existing val/test splits to output"""
    for split in ['val', 'test']:
        if structure['type'] == 'split':
            src_img = images_path / split
            src_lbl = labels_path / split
        elif structure['type'] == 'split_alt':
            src_img = images_path / split / 'images'
            src_lbl = labels_path / split / 'labels'
        else:
            continue
            
        if src_img.exists() and src_lbl.exists():
            dst_img = output_path / 'images' / split
            dst_lbl = output_path / 'labels' / split
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)
            
            for img in src_img.glob('*.*'):
                shutil.copy(img, dst_img / img.name)
            for lbl in src_lbl.glob('*.txt'):
                shutil.copy(lbl, dst_lbl / lbl.name)
            print(f"   Copied existing {split} split")

def main():
    print_banner("Image Augmentation")
    
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
    
    # Get target limit
    try:
        target = int(input(f"Enter target number of augmented images (default: {DEFAULT_AUG_TARGET}): ").strip() or str(DEFAULT_AUG_TARGET))
    except ValueError:
        target = DEFAULT_AUG_TARGET
    
    # Get output folder (will be created in Augmented directory)
    ensure_augmented_base()
    print(f"\n📁 Output will be saved in: {AUGMENTED_BASE}")
    
    default_name = f"{input_path.name}_augmented"
    print(f"   Default name: {default_name}")
    
    folder_name = input("Enter name for augmented dataset folder: ").strip()
    if not folder_name:
        folder_name = default_name
    
    output_path = AUGMENTED_BASE / folder_name
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output folder: {output_path}")
    
    # CRITICAL: Copy data.yaml to preserve class information
    print("\n📋 Preserving dataset configuration...")
    if not copy_data_yaml(input_path, output_path):
        print("   ⚠️ No data.yaml found, will create default after augmentation")
    
    print(f"\n🚀 Starting augmentation...")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Target: {target} images")
    
    # Get paths based on structure
    if structure['type'] == 'split':
        images_path = structure['images_path']
        labels_path = structure['labels_path']
    elif structure['type'] == 'split_alt':
        images_path = structure['images_path']
        labels_path = structure['labels_path']
    else:
        images_path = structure['images_path']
        labels_path = structure['labels_path']
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Copy existing val/test splits
    copy_existing_splits(output_path, images_path, labels_path, structure)
    
    # Get training images
    train_images = get_train_images(structure, images_path, labels_path)
    
    if not train_images:
        print("❌ No training images found!")
        return
    
    print(f"\n📸 Found {len(train_images)} source training images")
    
    out_img_dir = output_path / 'images' / 'train'
    out_lbl_dir = output_path / 'labels' / 'train'
    
    augmentations = ['flipH', 'flipV', 'dark', 'bright', 'gray', 'blur', 'contrast']
    total_saved = 0
    
    for img_path in train_images:
        if total_saved >= target:
            break
        
        stem = img_path.stem
        
        # Find label path based on structure
        if structure['type'] == 'split':
            lbl_path = labels_path / 'train' / f"{stem}.txt"
        elif structure['type'] == 'split_alt':
            lbl_path = labels_path / 'train' / 'labels' / f"{stem}.txt"
        else:
            lbl_path = labels_path / f"{stem}.txt"
        
        if not lbl_path.exists():
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        with open(lbl_path, 'r') as f:
            label_lines = [l.strip() for l in f.readlines() if l.strip()]
        
        if not label_lines:
            continue
        
        # Save original
        shutil.copy(img_path, out_img_dir / f"{stem}_orig.jpg")
        shutil.copy(lbl_path, out_lbl_dir / f"{stem}_orig.txt")
        total_saved += 1
        
        # Save augmented versions
        for aug_type in augmentations:
            if total_saved >= target:
                break
            
            aug_img, aug_labels = augment_image(img, label_lines, aug_type)
            out_path = out_img_dir / f"{stem}_{aug_type}.jpg"
            cv2.imwrite(str(out_path), aug_img)
            
            out_lbl_path = out_lbl_dir / f"{stem}_{aug_type}.txt"
            with open(out_lbl_path, 'w') as f:
                f.write("\n".join(aug_labels) + "\n")
            
            total_saved += 1
            
            if total_saved % 100 == 0:
                print(f"   Progress: {total_saved}/{target} images")
    
    print(f"\n✅ Augmentation complete! Saved {total_saved} images")
    print(f"   Output: {output_path}")

if __name__ == '__main__':
    main()