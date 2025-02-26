import os
import shutil
import random

# Full paths to directories
images_dir = 'C:/Users/Dell/Desktop/NASA_LAC/all_images/grayscale'
labels_dir = 'C:/Users/Dell/Desktop/NASA_LAC/output_labels'
train_images_dir = 'C:/Users/Dell/Desktop/NASA_LAC/dataset/images/train'
val_images_dir = 'C:/Users/Dell/Desktop/NASA_LAC/dataset/images/val'
train_labels_dir = 'C:/Users/Dell/Desktop/NASA_LAC/dataset/labels/train'
val_labels_dir = 'C:/Users/Dell/Desktop/NASA_LAC/dataset/labels/val'

# Create directories
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Get all image prefixes
image_files = [f for f in os.listdir(images_dir) if f.endswith('grayscale.png')]
prefixes = [f.split('grayscale')[0] for f in image_files]

# Ensure enough images exist for splitting
if len(prefixes) < 10:
    raise ValueError("Not enough images for proper dataset splitting!")

# Shuffle and split prefixes (80% train, 20% val)
random.shuffle(prefixes)
split_index = int(0.8 * len(prefixes))
train_prefixes = prefixes[:split_index]
val_prefixes = prefixes[split_index:]

# Move files to train and val directories
for prefix in train_prefixes:
    shutil.move(os.path.join(images_dir, f"{prefix}grayscale.png"), train_images_dir)
    if os.path.exists(os.path.join(labels_dir, f"{prefix}grayscale.txt")):
        shutil.move(os.path.join(labels_dir, f"{prefix}grayscale.txt"), train_labels_dir)

for prefix in val_prefixes:
    shutil.move(os.path.join(images_dir, f"{prefix}grayscale.png"), val_images_dir)
    if os.path.exists(os.path.join(labels_dir, f"{prefix}grayscale.txt")):
        shutil.move(os.path.join(labels_dir, f"{prefix}grayscale.txt"), val_labels_dir)

print("Dataset split complete! Training: {} images, Validation: {} images".format(len(train_prefixes), len(val_prefixes)))
