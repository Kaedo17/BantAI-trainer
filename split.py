#!/usr/bin/env python3
"""
Dataset splitting for YOLO format with automatic structure detection
Output saved to Output/Split/ folder
"""

import random
import shutil
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import SPLIT_RATIOS, print_banner
from utils import (
    get_folder_path, detect_folder_structure,
    validate_and_show_structure, normalize_to_yolo_format, update_data_yaml,
    ROOT_DIR
)

# Define split base directory
SPLIT_BASE = ROOT_DIR / "Output" / "Split"

def ensure_split_base():
    """Ensure the Split directory exists"""
    SPLIT_BASE.mkdir(parents=True, exist_ok=True)
    return SPLIT_BASE

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
    
    # Check if already has train/val/test
    if structure['has_train_val_test']:
        print("\n⚠️  This folder already has train/val/test splits!")
        choice = input("   Do you want to re-split anyway? (y/n): ").strip().lower()
        if choice != 'y':
            print("   Exiting without changes.")
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