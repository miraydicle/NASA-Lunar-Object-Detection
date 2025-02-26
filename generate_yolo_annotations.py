import os
import cv2
import numpy as np
from skimage.measure import label, regionprops

# Define full paths to directories
grayscale_dir = 'C:/Users/Dell/Desktop/NASA_LAC/all_images/grayscale'
semantic_dir = 'C:/Users/Dell/Desktop/NASA_LAC/all_images/semantic'
labels_dir = 'C:/Users/Dell/Desktop/NASA_LAC/output_labels'
dataset_yaml_path = 'C:/Users/Dell/Desktop/NASA_LAC/dataset.yaml'
os.makedirs(labels_dir, exist_ok=True)

# Predefined Object Classes (Fixed Mappings)
predefined_classes = {
    range(57, 62): 1,   # Rock
    range(136, 141): 2,  # Crater
    range(169, 173): 3,  # Lander
}

# Ignored Pixels (Background, unwanted colors)
ignore_pixels = {0, 33, 42}  # Background, pink, red (not objects)

# Dynamically assigned new class mappings
dynamic_class_mapping = {}  
next_dynamic_class_id = 4  # Start from 4 for newly detected colors

# Helper function to extract the prefix from file name
def extract_prefix(filename):
    return ''.join([c for c in filename if c.isdigit()])

# Function to get class ID based on pixel value
def get_class_id(pixel_value):
    global next_dynamic_class_id

    # Ensure the correct class is assigned to the lander
    if pixel_value == 171:
        return "lander"

    # Check predefined classes
    for pixel_range, class_name in predefined_classes.items():
        if pixel_value in pixel_range:
            return class_name  # Return predefined class

    # If pixel_value is not predefined, categorize it dynamically
    if pixel_value not in dynamic_class_mapping:
        dynamic_class_mapping[pixel_value] = f"class_{next_dynamic_class_id}"
        next_dynamic_class_id += 1  # Increment for next unknown class

    return dynamic_class_mapping[pixel_value]

# Pair grayscale and semantic images by prefix
grayscale_files = {extract_prefix(f): f for f in os.listdir(grayscale_dir) if f.endswith('grayscale.png')}
semantic_files = {extract_prefix(f): f for f in os.listdir(semantic_dir) if f.endswith('semantic.png')}

# Generate annotations
unique_classes = set(predefined_classes.values())  # Start with predefined classes
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

            # Remove ignored pixels (background, pink, red)
            mask_values = [val for val in mask_values if val not in ignore_pixels]

            # If no valid object is found, skip this region
            if len(mask_values) == 0:
                continue

            # Pick the most frequent object class in the bounding box
            most_frequent_value = int(np.bincount(mask_values).argmax())

            # Get class name using flexible range matching or dynamic assignment
            class_name = get_class_id(most_frequent_value)
            unique_classes.add(class_name)

            # Ensure small objects are detected by lowering threshold
            if (0.002 <= bbox_width <= 0.8 and 0.002 <= bbox_height <= 0.8 and
                0.01 <= x_center <= 0.99 and 0.01 <= y_center <= 0.99):
                f.write(f"{class_name} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

# Create final class list with both predefined and dynamically assigned classes
final_class_list = list(predefined_classes.values()) + list(dynamic_class_mapping.values())

# Update dataset.yaml with correct class count and names
nc = len(final_class_list)
dataset_yaml_content = f"""# Dataset path
path: C:/Users/Dell/Desktop/NASA_LAC/dataset

# Sub-paths for train and validation images
train: images/train
val: images/val

# Number of object classes
nc: {nc}

# Class names
names: {final_class_list}
"""

with open(dataset_yaml_path, "w") as f:
    f.write(dataset_yaml_content)

print(f"Annotation generation complete! dataset.yaml updated with {nc} classes: {final_class_list}")
