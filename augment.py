#!/usr/bin/env python3
"""
Image augmentation for YOLO datasets with automatic structure detection
Output saved to Output/Augmented/ folder with preserved class information
Includes automatic dataset cleaning to remove corrupt images
"""

import cv2
import numpy as np
import shutil
import yaml
from pathlib import Path
import sys
from tqdm import tqdm

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

def clean_dataset(dataset_path, output_path=None, remove_corrupt=True, remove_missing_labels=True):
    """
    Clean dataset by removing problematic images that cause training issues
    
    Args:
        dataset_path: Path to dataset to clean
        output_path: Path to save cleaned dataset (if None, overwrites)
        remove_corrupt: Remove images that can't be read
        remove_missing_labels: Remove images without labels
    
    Returns:
        tuple: (cleaned_dataset_path, stats)
    """
    dataset_path = Path(dataset_path)
    
    if output_path is None:
        output_path = dataset_path.parent / f"{dataset_path.name}_cleaned"
    
    print(f"\n🧹 Cleaning dataset: {dataset_path}")
    print(f"   Output: {output_path}")
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Copy data.yaml
    if (dataset_path / 'data.yaml').exists():
        shutil.copy(dataset_path / 'data.yaml', output_path / 'data.yaml')
    
    stats = {
        'total_removed': 0,
        'total_copied': 0,
        'corrupt_images': 0,
        'missing_labels': 0,
        'empty_labels': 0,
        'invalid_labels': 0,
        'by_split': {}
    }
    
    for split in ['train', 'val', 'test']:
        img_dir = dataset_path / 'images' / split
        lbl_dir = dataset_path / 'labels' / split
        
        if not img_dir.exists():
            continue
        
        images = list(img_dir.glob('*.*'))
        print(f"\n📁 Processing {split} split: {len(images)} images")
        
        split_stats = {
            'original': len(images),
            'removed': 0,
            'copied': 0
        }
        
        for img_path in tqdm(images, desc=f"Cleaning {split}"):
            valid = True
            reason = None
            
            # Check if image can be read
            if remove_corrupt:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        valid = False
                        reason = "cannot read"
                        stats['corrupt_images'] += 1
                    elif img.shape[0] == 0 or img.shape[1] == 0:
                        valid = False
                        reason = "zero dimension"
                        stats['corrupt_images'] += 1
                except Exception as e:
                    valid = False
                    reason = f"read error: {e}"
                    stats['corrupt_images'] += 1
            
            # Check for label file
            if valid and remove_missing_labels:
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    valid = False
                    reason = "missing label"
                    stats['missing_labels'] += 1
                else:
                    # Check label content
                    try:
                        with open(lbl_path, 'r') as f:
                            lines = [l.strip() for l in f.readlines() if l.strip()]
                            if not lines:
                                valid = False
                                reason = "empty label"
                                stats['empty_labels'] += 1
                            else:
                                # Check label format
                                for line in lines:
                                    parts = line.split()
                                    if len(parts) != 5:
                                        valid = False
                                        reason = "invalid format"
                                        stats['invalid_labels'] += 1
                                        break
                                    try:
                                        cls = int(parts[0])
                                        x, y, w, h = map(float, parts[1:5])
                                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                            valid = False
                                            reason = "coordinates out of range"
                                            stats['invalid_labels'] += 1
                                            break
                                    except:
                                        valid = False
                                        reason = "invalid values"
                                        stats['invalid_labels'] += 1
                                        break
                    except Exception as e:
                        valid = False
                        reason = f"label read error: {e}"
                        stats['invalid_labels'] += 1
            
            if valid:
                # Copy valid image and label
                shutil.copy(img_path, output_path / 'images' / split / img_path.name)
                if lbl_path.exists():
                    shutil.copy(lbl_path, output_path / 'labels' / split / lbl_path.name)
                stats['total_copied'] += 1
                split_stats['copied'] += 1
            else:
                # Remove problematic image
                stats['total_removed'] += 1
                split_stats['removed'] += 1
                if reason:
                    print(f"   ⚠️ Removed: {img_path.name} ({reason})")
        
        stats['by_split'][split] = split_stats
    
    print(f"\n✅ Cleaning complete!")
    print(f"   Total images processed: {stats['total_copied'] + stats['total_removed']}")
    print(f"   Valid images kept: {stats['total_copied']}")
    print(f"   Problematic images removed: {stats['total_removed']}")
    print(f"\n   Issues found:")
    print(f"      - Corrupt images: {stats['corrupt_images']}")
    print(f"      - Missing labels: {stats['missing_labels']}")
    print(f"      - Empty labels: {stats['empty_labels']}")
    print(f"      - Invalid labels: {stats['invalid_labels']}")
    
    return output_path, stats

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
    
    # Show source image count
    train_images = get_train_images(structure, 
        structure['images_path'] if structure['type'] in ['split', 'split_alt'] else structure['images_path'],
        structure['labels_path'] if structure['type'] in ['split', 'split_alt'] else structure['labels_path'])
    
    source_count = len(train_images)
    print(f"\n📸 Found {source_count} source training images")
    
    # Ask if user wants to clean the dataset first
    print("\n🧹 Dataset Cleaning Options:")
    print("   Cleaning removes corrupt images, missing labels, and invalid annotations.")
    print("   This prevents training errors and libpng warnings.")
    
    clean_choice = input("\nDo you want to clean the dataset before augmentation? (y/n, default: y): ").strip().lower()
    if clean_choice != 'n':
        print("\n🧹 Cleaning dataset...")
        cleaned_path, clean_stats = clean_dataset(input_path)
        if clean_stats['total_removed'] > 0:
            print(f"\n   ✅ Removed {clean_stats['total_removed']} problematic images")
            use_cleaned = input("\nUse cleaned dataset for augmentation? (y/n, default: y): ").strip().lower()
            if use_cleaned != 'n':
                input_path = cleaned_path
                print(f"\n📁 Using cleaned dataset: {input_path}")
                # Re-detect structure after cleaning
                structure = validate_and_show_structure(input_path)
    
    # Clear augmentation mode explanation
    print("\n" + "="*60)
    print("  AUGMENTATION MODES EXPLANATION")
    print("="*60)
    print("\nThis pipeline can create multiple variations of each image to increase dataset size.")
    print("\nThe available augmentations are:")
    print("   • Original (always included)")
    print("   • Horizontal Flip (mirror left-right)")
    print("   • Vertical Flip (mirror up-down)")
    print("   • Darken (reduce brightness)")
    print("   • Brighten (increase brightness)")
    print("   • Grayscale (convert to black & white)")
    print("   • Blur (Gaussian blur)")
    print("   • Contrast (enhance contrast)")
    
    print("\n" + "-"*60)
    print("  MODE 1: LIMIT TOTAL OUTPUT IMAGES")
    print("-"*60)
    print("Use this if you want to cap the total number of images after augmentation.")
    print("Example: You have 1,000 source images and set target to 3,000 total images.")
    print("Result: ~3 images created per source image (original + 2 variations).")
    print(f"Current source count: {source_count} images")
    
    print("\n" + "-"*60)
    print("  MODE 2: CREATE FIXED MULTIPLIER")
    print("-"*60)
    print("Use this to multiply your dataset by a specific factor.")
    print("Example: You have 1,000 source images and set multiplier to 8x.")
    print("Result: 8,000 total images (original + 7 variations per source).")
    print(f"Current source count: {source_count} images")
    print(f"Multiplier 8x would create: {source_count * 8} total images")
    
    print("\n" + "-"*60)
    print("  MODE 3: CREATE ALL VARIATIONS")
    print("-"*60)
    print("Use this to create ALL available augmentations for every image.")
    print("This creates 8 images per source (original + 7 augmentations).")
    print(f"Current source count: {source_count} images")
    print(f"All variations would create: {source_count * 8} total images")
    
    print("\n" + "="*60)
    print("  SELECT YOUR MODE")
    print("="*60)
    print("\n  1. Limit total output images (I want exactly X total images)")
    print("  2. Create fixed multiplier (I want X times more images)")
    print("  3. Create ALL variations (original + all 7 augmentations)")
    
    mode_choice = input("\n👉 Select mode (1, 2, or 3): ").strip()
    
    # Initialize variables
    target_total = None
    target_multiplier = None
    
    if mode_choice == '1':
        # Mode 1: Target total images
        print("\n" + "-"*60)
        print("  MODE 1: LIMIT TOTAL OUTPUT IMAGES")
        print("-"*60)
        print(f"Current source images: {source_count}")
        print("This mode will create approximately (target / source_count) images per source.")
        
        try:
            target_total = int(input(f"\nEnter target total number of images (minimum {source_count}): ").strip())
            if target_total < source_count:
                print(f"⚠️ Target cannot be less than source count ({source_count}). Using {source_count}.")
                target_total = source_count
        except ValueError:
            target_total = DEFAULT_AUG_TARGET
            print(f"Using default: {target_total}")
        
        # Calculate images per source
        images_per_source = max(1, target_total // source_count)
        actual_total = images_per_source * source_count
        print(f"\n📊 Results:")
        print(f"   → Will create {images_per_source} image(s) per source image")
        print(f"   → Estimated total: {actual_total} images")
        if actual_total < target_total:
            print(f"   → Note: Will create {actual_total} images (target was {target_total})")
        
    elif mode_choice == '2':
        # Mode 2: Fixed multiplier
        print("\n" + "-"*60)
        print("  MODE 2: CREATE FIXED MULTIPLIER")
        print("-"*60)
        print(f"Current source images: {source_count}")
        print("Multiplier 8x = original + 7 variations per image")
        print("Multiplier 4x = original + 3 variations per image")
        
        try:
            target_multiplier = int(input(f"\nEnter multiplier (2x to 8x, default: 8): ").strip() or "8")
            target_multiplier = max(2, min(8, target_multiplier))
        except ValueError:
            target_multiplier = 8
        
        images_per_source = target_multiplier
        actual_total = source_count * images_per_source
        
        print(f"\n📊 Results:")
        print(f"   → Will create {images_per_source} image(s) per source image")
        print(f"   → Original + {images_per_source - 1} variation(s)")
        print(f"   → Total: {actual_total} images")
        
    elif mode_choice == '3':
        # Mode 3: All variations
        print("\n" + "-"*60)
        print("  MODE 3: CREATE ALL VARIATIONS")
        print("-"*60)
        images_per_source = 8  # Original + 7 augmentations
        actual_total = source_count * images_per_source
        print(f"📊 Results:")
        print(f"   → Will create all 8 variations per image")
        print(f"   → Original + 7 augmentations")
        print(f"   → Total: {actual_total} images")
        
    else:
        print("\n❌ Invalid selection. Using MODE 1 with default target.")
        mode_choice = '1'
        target_total = DEFAULT_AUG_TARGET
        images_per_source = max(1, target_total // source_count)
        actual_total = images_per_source * source_count
    
    # Confirm before proceeding
    print("\n" + "="*60)
    print("  CONFIRMATION")
    print("="*60)
    print(f"\nSource images: {source_count}")
    print(f"Images per source: {images_per_source}")
    print(f"Total output images: {actual_total}")
    
    if mode_choice == '1':
        print(f"\nThis will create approximately {actual_total} images (target was {target_total})")
    else:
        print(f"\nThis will create {actual_total} images")
    
    confirm = input("\nProceed with augmentation? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Augmentation cancelled.")
        return
    
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
    
    # Get training images again (after structure detection)
    train_images = get_train_images(structure, images_path, labels_path)
    
    if not train_images:
        print("❌ No training images found!")
        return
    
    out_img_dir = output_path / 'images' / 'train'
    out_lbl_dir = output_path / 'labels' / 'train'
    
    # Available augmentations in order
    augmentations = ['flipH', 'flipV', 'dark', 'bright', 'gray', 'blur', 'contrast']
    
    # How many augmentations to apply per image
    num_augmentations = images_per_source - 1  # Subtract original
    
    print(f"\n🚀 Starting augmentation...")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Source images: {len(train_images)}")
    print(f"   Images per source: {images_per_source}")
    print(f"   Augmentations per image: {num_augmentations}")
    print(f"   Total target: {actual_total}")
    
    total_saved = 0
    corrupt_count = 0
    
    # Process each source image
    for idx, img_path in enumerate(train_images):
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
        
        # Try to read image - skip if corrupt
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                corrupt_count += 1
                if corrupt_count <= 10:  # Show first 10 corrupt images
                    print(f"   ⚠️ Skipping corrupt image: {img_path.name}")
                continue
            if img.shape[0] == 0 or img.shape[1] == 0:
                corrupt_count += 1
                continue
        except Exception as e:
            corrupt_count += 1
            continue
        
        # Read labels
        try:
            with open(lbl_path, 'r') as f:
                label_lines = [l.strip() for l in f.readlines() if l.strip()]
        except:
            continue
        
        if not label_lines:
            continue
        
        # Create variations for this image
        variations_created = 0
        
        # Always include original
        shutil.copy(img_path, out_img_dir / f"{stem}_orig.jpg")
        shutil.copy(lbl_path, out_lbl_dir / f"{stem}_orig.txt")
        variations_created += 1
        total_saved += 1
        
        # Create augmented versions
        aug_idx = 0
        while variations_created < images_per_source and aug_idx < len(augmentations):
            aug_type = augmentations[aug_idx]
            try:
                aug_img, aug_labels = augment_image(img, label_lines, aug_type)
                out_path = out_img_dir / f"{stem}_{aug_type}.jpg"
                cv2.imwrite(str(out_path), aug_img)
                
                out_lbl_path = out_lbl_dir / f"{stem}_{aug_type}.txt"
                with open(out_lbl_path, 'w') as f:
                    f.write("\n".join(aug_labels) + "\n")
                
                variations_created += 1
                total_saved += 1
            except Exception as e:
                pass  # Skip this augmentation if it fails
            aug_idx += 1
        
        # Progress update
        if (idx + 1) % 100 == 0:
            print(f"   Progress: {total_saved}/{actual_total} images ({idx + 1}/{len(train_images)} sources)")
    
    if corrupt_count > 0:
        print(f"\n   ⚠️ Skipped {corrupt_count} corrupt images during augmentation")
    
    print(f"\n✅ Augmentation complete!")
    print(f"   Output folder: {output_path}")
    print(f"\n📊 Final Summary:")
    print(f"   Source images processed: {len(train_images)}")
    print(f"   Total images created: {total_saved}")
    print(f"   Images per source average: {total_saved / len(train_images):.1f}")
    
    # Ask if user wants to clean the augmented dataset
    print("\n🧹 Final Cleanup:")
    print("   Cleaning removes any corrupt images created during augmentation.")
    clean_after = input("Do you want to clean the augmented dataset? (y/n, default: y): ").strip().lower()
    if clean_after != 'n':
        cleaned_path, clean_stats = clean_dataset(output_path)
        print(f"\n✅ Cleaned dataset saved to: {cleaned_path}")
        print(f"   Use this cleaned dataset for training to avoid errors!")

if __name__ == '__main__':
    main()