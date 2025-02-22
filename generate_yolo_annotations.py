import os
import cv2
import numpy as np
from skimage.measure import label, regionprops

# Define full paths to directories
grayscale_dir = 'C:/Users/Dell/Desktop/NASA_LAC/training_images/grayscale'
semantic_dir = 'C:/Users/Dell/Desktop/NASA_LAC/training_images/semantic'
labels_dir = 'C:/Users/Dell/Desktop/NASA_LAC/output_labels'
dataset_yaml_path = 'C:/Users/Dell/Desktop/NASA_LAC/dataset.yaml'
os.makedirs(labels_dir, exist_ok=True)

# Class Mapping
mask_class_mapping = {
    59: 1,   # Rock
    138: 2,  # Crater
    171: 3,  # Lunar Module
}

# Pixels to ignore
ignore_pixels = {0, 33, 42}  # Background, pink area, and non-object

# Helper function to extract the prefix from file name
def extract_prefix(filename):
    return ''.join([c for c in filename if c.isdigit()])

# Pair grayscale and semantic images by prefix
grayscale_files = {extract_prefix(f): f for f in os.listdir(grayscale_dir) if f.endswith('grayscale.png')}
semantic_files = {extract_prefix(f): f for f in os.listdir(semantic_dir) if f.endswith('semantic.png')}

# Generate annotations
unique_classes = set()
for prefix, grayscale_file in grayscale_files.items():
    if prefix not in semantic_files:
        print(f"[WARNING] No matching semantic mask for {grayscale_file}")
        continue

    # Read images
    grayscale_img = cv2.imread(os.path.join(grayscale_dir, grayscale_file), cv2.IMREAD_GRAYSCALE)
    semantic_mask = cv2.imread(os.path.join(semantic_dir, semantic_files[prefix]), cv2.IMREAD_GRAYSCALE)

    if grayscale_img is None or semantic_mask is None:
        print(f"[ERROR] Failed to read {prefix}. Skipping.")
        continue

    height, width = grayscale_img.shape

    # Label connected regions in the mask
    labeled_mask = label(semantic_mask)

    # Create YOLO annotation file
    annotation_file = os.path.join(labels_dir, f"{prefix}grayscale.txt")
    with open(annotation_file, 'w') as f:
        for region in regionprops(labeled_mask):
            # Get bounding box coordinates
            minr, minc, maxr, maxc = region.bbox
            x_center = (minc + maxc) / 2 / width
            y_center = (minr + maxr) / 2 / height
            bbox_width = (maxc - minc) / width
            bbox_height = (maxr - minr) / height

            # Extract the unique values from the mask region
            mask_values = np.unique(semantic_mask[minr:maxr, minc:maxc])

            # Remove ignored pixels (background, pink, non-object)
            mask_values = [val for val in mask_values if val not in ignore_pixels]

            # If no valid object is found, skip this region
            if not mask_values:
                continue

            # Pick the most frequent object class in the bounding box
            class_counts = np.bincount(mask_values)
            most_frequent_value = mask_values[np.argmax(class_counts[:len(mask_values)])]

            # Ensure the class ID is correctly mapped
            class_id = mask_class_mapping.get(most_frequent_value, 4)  # Default to "unknown_object"

            unique_classes.add(class_id)

            # Ensure valid bounding boxes (filter tiny boxes, edge cases, and large boxes)
            if (0.02 <= bbox_width <= 0.8 and 0.02 <= bbox_height <= 0.8 and
                0.02 <= x_center <= 0.98 and 0.02 <= y_center <= 0.98):
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

# Class Names
class_names = ["rock", "crater", "lunar_module", "unknown_object"]

# Update dataset.yaml with correct class count and names
nc = len(class_names)
dataset_yaml_content = f"""# Dataset path
path: C:/Users/Dell/Desktop/NASA_LAC/dataset

# Sub-paths for train and validation images
train: images/train
val: images/val

# Number of object classes
nc: {nc}

# Class names
names: {class_names}
"""

with open(dataset_yaml_path, "w") as f:
    f.write(dataset_yaml_content)

print(f"Annotation generation complete! dataset.yaml updated with {nc} classes: {class_names}")
