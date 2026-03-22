#!/usr/bin/env python3
"""
Image augmentation for YOLO datasets
"""

import cv2
import numpy as np
from pathlib import Path
import sys
from config import DEFAULT_AUG_TARGET, print_banner
from utils import get_folder_path, get_output_path, validate_yolo_structure

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

def main():
    print_banner("Image Augmentation")
    
    # Get input folder
    print("\n📁 Your folder should have 'images/train' and 'labels/train'")
    input_path = get_folder_path("Enter input folder path")
    
    # Validate structure
    valid, images_path, labels_path = validate_yolo_structure(input_path)
    if not valid:
        print("❌ Invalid folder structure. Must have images/train and labels/train")
        sys.exit(1)
    
    # Get output folder
    default_output = get_output_path(input_path, "augmented")
    print(f"\n💡 Suggested output: {default_output}")
    output_path = get_folder_path("Enter output folder path (or press Enter for suggested)")
    
    # Get target limit
    try:
        target = int(input(f"Enter target number of augmented images (default: {DEFAULT_AUG_TARGET}): ").strip() or str(DEFAULT_AUG_TARGET))
    except ValueError:
        target = DEFAULT_AUG_TARGET
    
    print(f"\n🚀 Starting augmentation...")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Target: {target} images")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Copy existing val/test if they exist
    for split in ['val', 'test']:
        src_img = images_path / split
        src_lbl = labels_path / split
        if src_img.exists() and src_lbl.exists():
            for img in src_img.glob('*.*'):
                shutil.copy(img, output_path / 'images' / split / img.name)
            for lbl in src_lbl.glob('*.txt'):
                shutil.copy(lbl, output_path / 'labels' / split / lbl.name)
            print(f"   Copied existing {split} split")
    
    # Augment training images
    train_img_path = images_path / 'train'
    train_lbl_path = labels_path / 'train'
    
    if not train_img_path.exists():
        print("❌ No 'train' folder found!")
        sys.exit(1)
    
    # Find all images
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        images.extend(list(train_img_path.glob(ext)))
    
    print(f"\n📸 Found {len(images)} source training images")
    
    if len(images) == 0:
        print("❌ No images found!")
        sys.exit(1)
    
    out_img_dir = output_path / 'images' / 'train'
    out_lbl_dir = output_path / 'labels' / 'train'
    
    augmentations = ['flipH', 'flipV', 'dark', 'bright', 'gray', 'blur', 'contrast']
    total_saved = 0
    
    for img_path in images:
        if total_saved >= target:
            break
        
        stem = img_path.stem
        lbl_path = train_lbl_path / f"{stem}.txt"
        
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