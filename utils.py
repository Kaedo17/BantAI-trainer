#!/usr/bin/env python3
"""
Utility functions for YOLO pipeline with automatic structure detection
"""

import os
import yaml
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def get_folder_path(prompt):
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

def get_output_path(input_path, suffix):
    """Generate output path based on input folder name"""
    parent = input_path.parent
    name = f"{input_path.name}_{suffix}"
    return parent / name

def detect_folder_structure(folder_path: Path) -> Dict:
    """
    Automatically detect YOLO folder structure
    
    Returns a dictionary with:
    - type: 'split' (has train/val/test subfolders) or 'flat' (all files together)
    - images_path: Path to images folder
    - labels_path: Path to labels folder
    - splits: List of available splits (for split structure)
    """
    result = {
        'type': 'unknown',
        'images_path': None,
        'labels_path': None,
        'splits': [],
        'has_train_val_test': False,
        'is_flat': False
    }
    
    # Check for common structures
    images_path = folder_path / 'images'
    labels_path = folder_path / 'labels'
    
    # Structure 1: folder/images/train, folder/images/val, folder/labels/train, folder/labels/val
    if images_path.exists() and labels_path.exists():
        # Check for subfolders
        splits = []
        for split in ['train', 'val', 'test']:
            if (images_path / split).exists() and (labels_path / split).exists():
                splits.append(split)
        
        if splits:
            result['type'] = 'split'
            result['images_path'] = images_path
            result['labels_path'] = labels_path
            result['splits'] = splits
            result['has_train_val_test'] = True
            return result
    
    # Structure 2: folder/train/images, folder/val/images, folder/train/labels, folder/val/labels
    train_images = folder_path / 'train' / 'images'
    train_labels = folder_path / 'train' / 'labels'
    val_images = folder_path / 'val' / 'images'
    val_labels = folder_path / 'val' / 'labels'
    
    if train_images.exists() and train_labels.exists():
        result['type'] = 'split_alt'
        result['images_path'] = folder_path
        result['labels_path'] = folder_path
        result['splits'] = []
        if train_images.exists():
            result['splits'].append('train')
        if val_images.exists():
            result['splits'].append('val')
        if (folder_path / 'test' / 'images').exists():
            result['splits'].append('test')
        result['has_train_val_test'] = True
        return result
    
    # Structure 3: folder/images and folder/labels with all files (flat)
    if images_path.exists() and labels_path.exists():
        # Check if there are files directly in images folder (not subfolders)
        images_files = list(images_path.glob('*.*'))
        if len(images_files) > 0:
            result['type'] = 'flat'
            result['images_path'] = images_path
            result['labels_path'] = labels_path
            result['is_flat'] = True
            return result
    
    # Structure 4: folder/ (images and labels mixed together)
    mixed_files = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))
    if len(mixed_files) > 0:
        # Check if there are corresponding label files
        has_labels = any((folder_path / f"{f.stem}.txt").exists() for f in mixed_files)
        if has_labels:
            result['type'] = 'mixed'
            result['images_path'] = folder_path
            result['labels_path'] = folder_path
            result['is_flat'] = True
            return result
    
    return result

def validate_and_show_structure(folder_path: Path) -> Dict:
    """Validate folder and show detected structure"""
    structure = detect_folder_structure(folder_path)
    
    print(f"\n📁 Detected structure: {structure['type'].upper()}")
    
    if structure['type'] == 'split':
        print(f"   Splits found: {', '.join(structure['splits'])}")
        for split in structure['splits']:
            img_count = len(list((structure['images_path'] / split).glob('*.*')))
            lbl_count = len(list((structure['labels_path'] / split).glob('*.txt')))
            print(f"   {split}: {img_count} images, {lbl_count} labels")
            
    elif structure['type'] == 'split_alt':
        print(f"   Alternative split structure detected")
        for split in structure['splits']:
            img_count = len(list((structure['images_path'] / split / 'images').glob('*.*')))
            lbl_count = len(list((structure['labels_path'] / split / 'labels').glob('*.txt')))
            print(f"   {split}: {img_count} images, {lbl_count} labels")
            
    elif structure['type'] == 'flat':
        img_count = len(list(structure['images_path'].glob('*.*')))
        lbl_count = len(list(structure['labels_path'].glob('*.txt')))
        print(f"   Images: {img_count}")
        print(f"   Labels: {lbl_count}")
        
    elif structure['type'] == 'mixed':
        img_count = len(list(structure['images_path'].glob('*.jpg')) + 
                       list(structure['images_path'].glob('*.png')))
        lbl_count = len(list(structure['labels_path'].glob('*.txt')))
        print(f"   Images: {img_count}")
        print(f"   Labels: {lbl_count}")
    
    return structure

def copy_files_with_structure(src_path: Path, dst_path: Path, files_dict: Dict[str, List[Path]]):
    """Copy files preserving structure"""
    for split_name, images_list in files_dict.items():
        for img in images_list:
            # Copy image
            shutil.copy(img, dst_path / 'images' / split_name / img.name)
            # Copy corresponding label
            lbl = img.with_suffix('.txt')
            if lbl.exists():
                shutil.copy(lbl, dst_path / 'labels' / split_name / lbl.name)

def update_data_yaml(folder_path: Path, class_names: List[str]):
    """Create or update data.yaml file"""
    data_yaml = {
        'path': str(folder_path),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = folder_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    return yaml_path

def get_class_names_from_labels(labels_path: Path, images_list: List[Path]) -> List[str]:
    """Extract unique class names from label files"""
    class_ids = set()
    class_names_dict = {}
    
    for img in images_list:
        lbl = labels_path / f"{img.stem}.txt"
        if lbl.exists():
            with open(lbl, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            class_ids.add(int(line.split()[0]))
                        except:
                            pass
    
    # Create class names (use actual names if available)
    class_names = [f"class_{i}" for i in sorted(class_ids)]
    return class_names

def get_file_path(prompt, extension=".pt"):
    """Get file path from user with validation"""
    while True:
        path = input(f"{prompt}: ").strip().strip('"')
        if not path:
            print("❌ Path cannot be empty. Please try again.")
            continue
        path = Path(path)
        if not path.exists():
            print(f"❌ Path '{path}' does not exist. Please try again.")
            continue
        if not path.is_file():
            print(f"❌ '{path}' is not a file. Please try again.")
            continue
        if extension and path.suffix != extension:
            print(f"❌ Please select a {extension} file. Got: {path.suffix}")
            continue
        return path

def get_folder_path(prompt):
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

def normalize_to_yolo_format(input_path: Path, output_path: Path, split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
    """
    Convert any detected structure to standard YOLO format:
    output_path/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
    """
    structure = detect_folder_structure(input_path)
    
    if structure['type'] == 'unknown':
        print("❌ Could not detect folder structure")
        return False, None
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Collect all images based on structure
    all_images = []
    
    if structure['type'] == 'split':
        # Already split structure - combine all splits first if needed
        for split in structure['splits']:
            img_path = structure['images_path'] / split
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                all_images.extend(list(img_path.glob(ext)))
    
    elif structure['type'] == 'split_alt':
        # Alternative split structure
        for split in structure['splits']:
            img_path = structure['images_path'] / split / 'images'
            if img_path.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    all_images.extend(list(img_path.glob(ext)))
    
    elif structure['type'] in ['flat', 'mixed']:
        # Flat structure - all images in one folder
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            all_images.extend(list(structure['images_path'].glob(ext)))
    
    if not all_images:
        print("❌ No images found")
        return False, None
    
    # Filter images that have labels
    valid_images = []
    for img in all_images:
        lbl_path = structure['labels_path'] / f"{img.stem}.txt"
        if lbl_path.exists():
            valid_images.append(img)
    
    print(f"   Found {len(valid_images)} images with labels")
    
    if len(valid_images) == 0:
        print("❌ No images with matching labels found")
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
        for img in images_list:
            # Copy image
            shutil.copy(img, output_path / 'images' / split_name / img.name)
            # Copy label
            lbl = structure['labels_path'] / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy(lbl, output_path / 'labels' / split_name / lbl.name)
    
    # Get class names
    class_names = get_class_names_from_labels(structure['labels_path'], valid_images)
    
    # Create data.yaml
    yaml_path = update_data_yaml(output_path, class_names)
    
    return True, yaml_path