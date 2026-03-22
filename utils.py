#!/usr/bin/env python3
"""
Utility functions for YOLO pipeline
"""

import os
import yaml
import shutil
import random
from pathlib import Path

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

def validate_yolo_structure(folder_path):
    """Check if folder has valid YOLO structure with train/val/test splits"""
    images_path = folder_path / 'images'
    labels_path = folder_path / 'labels'
    
    if not images_path.exists() or not labels_path.exists():
        return False, None, None
    
    splits = ['train', 'val', 'test']
    found_splits = []
    
    for split in splits:
        if (images_path / split).exists() and (labels_path / split).exists():
            found_splits.append(split)
    
    return len(found_splits) > 0, images_path, labels_path

def update_data_yaml(folder_path, class_names):
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