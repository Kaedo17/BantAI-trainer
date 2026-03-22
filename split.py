#!/usr/bin/env python3
"""
Dataset splitting for YOLO format with automatic structure detection
"""

import random
import shutil
from pathlib import Path
import sys
from config import SPLIT_RATIOS, print_banner
from utils import (
    get_folder_path, get_output_path, detect_folder_structure,
    validate_and_show_structure, normalize_to_yolo_format, update_data_yaml
)

def main():
    print_banner("Dataset Splitting (80% Train, 10% Val, 10% Test)")
    
    # Get input folder
    input_path = get_folder_path("Enter input folder path")
    
    # Detect and show structure
    print("\n🔍 Analyzing folder structure...")
    structure = validate_and_show_structure(input_path)
    
    if structure['type'] == 'unknown':
        print("❌ Could not detect folder structure")
        print("   Expected structures:")
        print("   1. folder/images/train, folder/images/val, folder/labels/train, folder/labels/val")
        print("   2. folder/train/images, folder/val/images, folder/train/labels, folder/val/labels")
        print("   3. folder/images, folder/labels (flat structure)")
        sys.exit(1)
    
    # Check if already has train/val/test
    if structure['has_train_val_test']:
        print("\n⚠️  This folder already has train/val/test splits!")
        choice = input("   Do you want to re-split anyway? (y/n): ").strip().lower()
        if choice != 'y':
            print("   Exiting without changes.")
            sys.exit(0)
    
    # Get output folder
    default_output = get_output_path(input_path, "split")
    print(f"\n💡 Suggested output: {default_output}")
    output_path = get_folder_path("Enter output folder path (or press Enter for suggested)")
    
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
            print(f"   images/{split}: {img_count} files")
            print(f"   labels/{split}: {lbl_count} files")
    else:
        print("❌ Failed to split dataset")

if __name__ == '__main__':
    main()