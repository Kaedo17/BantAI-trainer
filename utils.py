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

# Global root directory for outputs
ROOT_DIR = Path.cwd()
OUTPUT_BASE = ROOT_DIR / "Output"

def ensure_output_base():
    """Ensure the base output directory exists"""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    return OUTPUT_BASE

def get_folder_path(prompt, allow_cancel=True):
    """Get folder path from user with validation and cancel option"""
    while True:
        if allow_cancel:
            print("   (Type 'back' to go back, 'cancel' to exit)")
        path = input(f"{prompt}: ").strip().strip('"')
        
        if allow_cancel:
            if path.lower() == 'back':
                return 'BACK'
            if path.lower() == 'cancel':
                return 'CANCEL'
        
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

def get_folder_path_with_browse(prompt, allow_cancel=True):
    """Get folder path with option to browse current directory"""
    while True:
        print(f"\n📁 {prompt}")
        print("   Options:")
        print("      - Type a path")
        print("      - Type 'browse' to see current directory contents")
        if allow_cancel:
            print("      - Type 'back' to go back")
            print("      - Type 'cancel' to exit")
        
        choice = input("\n👉 Your choice: ").strip().strip('"')
        
        if choice.lower() == 'browse':
            current_dir = Path.cwd()
            print(f"\n📂 Current directory: {current_dir}")
            items = list(current_dir.iterdir())
            folders = [item for item in items if item.is_dir()]
            files = [item for item in items if item.is_file()]
            
            if folders:
                print("\n   📁 Folders:")
                for folder in sorted(folders)[:20]:
                    print(f"      {folder.name}")
            if files:
                print("\n   📄 Files (first 20):")
                for file in sorted(files)[:20]:
                    print(f"      {file.name}")
            print("\n   (Use full path to navigate to subfolders)")
            continue
        
        if allow_cancel:
            if choice.lower() == 'back':
                return 'BACK'
            if choice.lower() == 'cancel':
                return 'CANCEL'
        
        if not choice:
            print("❌ Path cannot be empty. Please try again.")
            continue
        
        path = Path(choice)
        if not path.exists():
            print(f"❌ Path '{path}' does not exist. Please try again.")
            continue
        if not path.is_dir():
            print(f"❌ '{path}' is not a directory. Please try again.")
            continue
        return path

def get_output_folder(prompt, default_name=None, allow_cancel=True):
    """Get output folder path with creation option"""
    ensure_output_base()
    
    while True:
        print(f"\n📁 {prompt}")
        if default_name:
            print(f"   Default: {OUTPUT_BASE / default_name}")
        print("   Options:")
        print("      - Enter a folder name (will be created in Output folder)")
        print("      - Type a full path")
        print("      - Type 'default' to use default")
        if allow_cancel:
            print("      - Type 'back' to go back")
            print("      - Type 'cancel' to exit")
        
        choice = input("\n👉 Folder name/path: ").strip().strip('"')
        
        if allow_cancel:
            if choice.lower() == 'back':
                return 'BACK'
            if choice.lower() == 'cancel':
                return 'CANCEL'
        
        if choice.lower() == 'default' and default_name:
            output_path = OUTPUT_BASE / default_name
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Using default: {output_path}")
            return output_path
        
        if not choice:
            print("❌ Cannot be empty. Please try again.")
            continue
        
        # Check if it's a simple name or a path
        if '/' not in choice and '\\' not in choice and not Path(choice).parent != Path(choice):
            # Simple folder name - create in Output directory
            output_path = OUTPUT_BASE / choice
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created folder: {output_path}")
            return output_path
        else:
            # Full path
            output_path = Path(choice)
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created/Using folder: {output_path}")
                return output_path
            except Exception as e:
                print(f"❌ Could not create folder: {e}")
                continue

def get_output_path(input_path, suffix):
    """Generate output path based on input folder name (legacy)"""
    parent = input_path.parent
    name = f"{input_path.name}_{suffix}"
    return parent / name

def detect_folder_structure(folder_path: Path) -> Dict:
    """Automatically detect YOLO folder structure"""
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
            shutil.copy(img, dst_path / 'images' / split_name / img.name)
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
    
    class_names = [f"class_{i}" for i in sorted(class_ids)]
    return class_names

def get_file_path(prompt, extension=".pt", allow_cancel=True):
    """Get file path from user with validation and cancel option"""
    while True:
        if allow_cancel:
            print("   (Type 'back' to go back, 'cancel' to exit)")
        path = input(f"{prompt}: ").strip().strip('"')
        
        if allow_cancel:
            if path.lower() == 'back':
                return 'BACK'
            if path.lower() == 'cancel':
                return 'CANCEL'
        
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

def normalize_to_yolo_format(input_path: Path, output_path: Path, split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
    """Convert any detected structure to standard YOLO format"""
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
        for split in structure['splits']:
            img_path = structure['images_path'] / split
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
            shutil.copy(img, output_path / 'images' / split_name / img.name)
            lbl = structure['labels_path'] / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy(lbl, output_path / 'labels' / split_name / lbl.name)
    
    # Get class names
    class_names = get_class_names_from_labels(structure['labels_path'], valid_images)
    
    # Create data.yaml
    yaml_path = update_data_yaml(output_path, class_names)
    
    return True, yaml_path