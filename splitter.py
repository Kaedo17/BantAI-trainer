import os
import shutil
import random
from pathlib import Path

# Setup paths
dataset_path = Path("archive/Open-Flame-Hazards (1)")
output_path = Path("archive/Open-Flame-Hazards-YOLO")

# Create train/val folders
for split in ['train', 'val']:
    (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
    (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

# Get all images
images = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png")) + list(dataset_path.glob("*.jpeg"))

# Split 80/20
random.shuffle(images)
split_idx = int(len(images) * 0.8)
train_images = images[:split_idx]
val_images = images[split_idx:]

# Copy files
for img in train_images:
    shutil.copy(img, output_path / 'train' / 'images' / img.name)
    label = img.with_suffix('.txt')
    if label.exists():
        shutil.copy(label, output_path / 'train' / 'labels' / label.name)

for img in val_images:
    shutil.copy(img, output_path / 'val' / 'images' / img.name)
    label = img.with_suffix('.txt')
    if label.exists():
        shutil.copy(label, output_path / 'val' / 'labels' / label.name)

# Create data.yaml for the new structure
yaml_content = f"""path: {output_path}
train: train/images
val: val/images

nc: 1
names: ['Open-Flame-Hazard']
"""

with open(output_path / 'data.yaml', 'w') as f:
    f.write(yaml_content)

print(f"✅ Dataset prepared at: {output_path}")
print(f"Train images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print(f"Classes: 1 (Open-Flame-Hazard)")