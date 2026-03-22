#!/usr/bin/env python3
"""
Dataset Merger - Merge multiple YOLO datasets into one unified dataset
Supports merging datasets with automatic structure detection and class mapping
"""

import os
import shutil
import yaml
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import print_banner
from utils import detect_folder_structure, validate_and_show_structure


class DatasetMerger:
    """Merge multiple YOLO datasets into one"""
    
    def __init__(self):
        self.datasets = []
        self.class_mappings = {}
        self.global_class_map = {}
        self.next_class_id = 0
        
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
    
    def detect_dataset_info(self, dataset_path: Path) -> Dict:
        """Detect dataset information including classes"""
        structure = detect_folder_structure(dataset_path)
        
        if structure['type'] == 'unknown':
            print(f"❌ Could not detect structure for {dataset_path}")
            return None
        
        print(f"\n📁 Analyzing: {dataset_path.name}")
        print(f"   Structure: {structure['type'].upper()}")
        
        # Find all images and extract class IDs
        images = []
        labels_path = None
        
        if structure['type'] == 'split':
            # Get from train split
            train_img_path = structure['images_path'] / 'train'
            train_lbl_path = structure['labels_path'] / 'train'
            
            if train_img_path.exists() and train_lbl_path.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    images.extend(list(train_img_path.glob(ext)))
                labels_path = train_lbl_path
                
        elif structure['type'] == 'split_alt':
            train_img_path = structure['images_path'] / 'train' / 'images'
            train_lbl_path = structure['labels_path'] / 'train' / 'labels'
            
            if train_img_path.exists() and train_lbl_path.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    images.extend(list(train_img_path.glob(ext)))
                labels_path = train_lbl_path
                
        elif structure['type'] in ['flat', 'mixed']:
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                images.extend(list(structure['images_path'].glob(ext)))
            labels_path = structure['labels_path']
        
        if not images or not labels_path:
            print(f"   ⚠️ No images found in {dataset_path}")
            return None
        
        # Extract class IDs from labels
        class_ids = set()
        class_names = {}
        
        # Check for existing data.yaml
        data_yaml = dataset_path / 'data.yaml'
        if data_yaml.exists():
            try:
                with open(data_yaml, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    if 'names' in yaml_data:
                        class_names = {i: name for i, name in enumerate(yaml_data['names'])}
            except:
                pass
        
        # Scan labels to find class IDs
        for img in images[:100]:  # Sample first 100 images
            lbl = labels_path / f"{img.stem}.txt"
            if lbl.exists():
                with open(lbl, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                cls_id = int(line.split()[0])
                                class_ids.add(cls_id)
                            except:
                                pass
        
        print(f"   Found {len(images)} images")
        print(f"   Found {len(class_ids)} unique class IDs: {sorted(class_ids)}")
        
        # Get class names
        if class_names:
            print(f"   Class names from data.yaml: {class_names}")
        else:
            class_names = {cid: f"class_{cid}" for cid in sorted(class_ids)}
            print(f"   Using generated class names: {class_names}")
        
        return {
            'path': dataset_path,
            'structure': structure,
            'images': images,
            'labels_path': labels_path,
            'class_ids': class_ids,
            'class_names': class_names,
            'name': dataset_path.name
        }
    
    def map_classes(self, dataset_info: Dict, dataset_index: int):
        """Map local class IDs to global class IDs"""
        local_classes = dataset_info['class_ids']
        local_names = dataset_info['class_names']
        
        print(f"\n📋 Mapping classes for dataset {dataset_index + 1}: {dataset_info['name']}")
        
        mapping = {}
        for local_id in sorted(local_classes):
            local_name = local_names.get(local_id, f"class_{local_id}")
            
            # Check if this class already exists in global map
            global_id = None
            for gid, gname in self.global_class_map.items():
                if gname.lower() == local_name.lower():
                    global_id = gid
                    print(f"   '{local_name}' (local ID {local_id}) → global ID {global_id} (matched by name)")
                    break
            
            if global_id is None:
                global_id = self.next_class_id
                self.global_class_map[global_id] = local_name
                self.next_class_id += 1
                print(f"   '{local_name}' (local ID {local_id}) → new global ID {global_id}")
            
            mapping[local_id] = global_id
        
        return mapping
    
    def update_labels(self, source_dir: Path, labels_path: Path, mapping: Dict, output_labels_dir: Path):
        """Copy and update labels with new class IDs"""
        count = 0
        
        for img in source_dir.glob('*.*'):
            if img.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            lbl_file = labels_path / f"{img.stem}.txt"
            if not lbl_file.exists():
                continue
            
            output_lbl = output_labels_dir / f"{img.stem}.txt"
            
            # Read and update labels
            updated_lines = []
            with open(lbl_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        try:
                            old_class = int(parts[0])
                            new_class = mapping.get(old_class)
                            if new_class is not None:
                                parts[0] = str(new_class)
                                updated_lines.append(' '.join(parts))
                            else:
                                print(f"   ⚠️ Warning: Class {old_class} not mapped in {lbl_file}")
                        except:
                            updated_lines.append(line)
            
            # Write updated labels
            if updated_lines:
                with open(output_lbl, 'w') as f:
                    f.write('\n'.join(updated_lines) + '\n')
                count += 1
        
        return count
    
    def merge_datasets(self):
        """Main merge function"""
        print_banner("Dataset Merger")
        
        # Get number of datasets to merge
        try:
            num_datasets = int(input("\n📊 How many datasets do you want to merge? ").strip())
        except ValueError:
            print("❌ Please enter a valid number")
            return
        
        # Collect all datasets
        print("\n📁 Please provide the paths to your datasets:")
        print("   (Each dataset should have images and labels in YOLO format)")
        
        for i in range(num_datasets):
            print(f"\n--- Dataset {i+1} ---")
            dataset_path = self.get_folder_path(f"Enter path to dataset {i+1}")
            
            info = self.detect_dataset_info(dataset_path)
            if info:
                self.datasets.append(info)
            else:
                print(f"⚠️ Skipping dataset {i+1} - could not detect structure")
        
        if len(self.datasets) == 0:
            print("❌ No valid datasets found. Exiting.")
            return
        
        # Show summary of datasets
        print_banner("Datasets Summary")
        for i, ds in enumerate(self.datasets):
            print(f"\nDataset {i+1}: {ds['name']}")
            print(f"   Images: {len(ds['images'])}")
            print(f"   Classes: {ds['class_ids']}")
            print(f"   Class names: {ds['class_names']}")
        
        # Create class mappings
        print_banner("Class Mapping")
        for i, ds in enumerate(self.datasets):
            mapping = self.map_classes(ds, i)
            self.class_mappings[ds['path']] = mapping
        
        # Show final global class list
        print_banner("Final Global Classes")
        for class_id in sorted(self.global_class_map.keys()):
            print(f"   ID {class_id}: {self.global_class_map[class_id]}")
        print(f"\n   Total classes: {len(self.global_class_map)}")
        
        # Get output folder
        default_output = Path.cwd() / "merged_dataset"
        print(f"\n💡 Suggested output: {default_output}")
        output_path = self.get_folder_path("Enter output folder path (or press Enter for suggested)")
        
        # Create output structure
        for split in ['train', 'val', 'test']:
            (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Merge datasets
        print_banner("Merging Datasets")
        
        total_images = 0
        total_labels_updated = 0
        
        for ds in self.datasets:
            print(f"\n📦 Processing: {ds['name']}")
            
            # Determine source paths
            if ds['structure']['type'] == 'split':
                img_source = ds['structure']['images_path'] / 'train'
                lbl_source = ds['structure']['labels_path'] / 'train'
            elif ds['structure']['type'] == 'split_alt':
                img_source = ds['structure']['images_path'] / 'train' / 'images'
                lbl_source = ds['structure']['labels_path'] / 'train' / 'labels'
            else:
                img_source = ds['structure']['images_path']
                lbl_source = ds['structure']['labels_path']
            
            # Output paths (all go to train, we'll split later if needed)
            output_images = output_path / 'images' / 'train'
            output_labels = output_path / 'labels' / 'train'
            
            # Get mapping for this dataset
            mapping = self.class_mappings[ds['path']]
            
            # Copy and update images and labels
            images_copied = 0
            for img in img_source.glob('*.*'):
                if img.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue
                
                # Copy image with unique name to avoid conflicts
                new_name = f"{ds['name']}_{img.name}"
                shutil.copy(img, output_images / new_name)
                
                # Update and copy label
                lbl_file = lbl_source / f"{img.stem}.txt"
                if lbl_file.exists():
                    output_lbl = output_labels / f"{ds['name']}_{img.stem}.txt"
                    
                    updated_lines = []
                    with open(lbl_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split()
                                try:
                                    old_class = int(parts[0])
                                    new_class = mapping.get(old_class)
                                    if new_class is not None:
                                        parts[0] = str(new_class)
                                        updated_lines.append(' '.join(parts))
                                except:
                                    updated_lines.append(line)
                    
                    if updated_lines:
                        with open(output_lbl, 'w') as f:
                            f.write('\n'.join(updated_lines) + '\n')
                        images_copied += 1
                    else:
                        # No valid labels, remove the image we copied
                        (output_images / new_name).unlink()
            
            print(f"   Copied {images_copied} images with updated labels")
            total_images += images_copied
        
        print(f"\n✅ Total images merged: {total_images}")
        
        # Create data.yaml
        class_names_list = [self.global_class_map[i] for i in sorted(self.global_class_map.keys())]
        data_yaml = {
            'path': str(output_path),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names_list),
            'names': class_names_list
        }
        
        yaml_path = output_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"\n📄 Created data.yaml: {yaml_path}")
        print(f"   Classes: {len(class_names_list)}")
        print(f"   Class names: {class_names_list}")
        
        # Ask if user wants to split the merged dataset
        print_banner("Next Steps")
        print("\nYour merged dataset is ready in the 'train' folder.")
        print("You can now:")
        print("   1. Run the split script to create train/val/test splits")
        print("   2. Or use this dataset directly with 'Quick Train'")
        
        split_choice = input("\nWould you like to split this dataset now? (y/n): ").strip().lower()
        if split_choice == 'y':
            print("\n🚀 Running dataset split...")
            # Import and run split
            try:
                from split import main as split_main
                # Temporarily override sys.argv
                original_argv = sys.argv
                sys.argv = ['split.py']
                split_main()
                sys.argv = original_argv
            except ImportError:
                print("⚠️ Could not import split module. Please run split.py manually.")
            except Exception as e:
                print(f"⚠️ Split failed: {e}")
        
        print(f"\n✅ Merge complete! Output saved to: {output_path}")


def main():
    """Main entry point"""
    merger = DatasetMerger()
    merger.merge_datasets()


if __name__ == '__main__':
    main()