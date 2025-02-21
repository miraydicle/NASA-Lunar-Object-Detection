import os
import cv2
from skimage.measure import label, regionprops

# Define full paths to directories
grayscale_dir = 'C:/Users/Dell/Desktop/NASA_LAC/training_images/grayscale'
semantic_dir = 'C:/Users/Dell/Desktop/NASA_LAC/training_images/semantic'
labels_dir = 'C:/Users/Dell/Desktop/NASA_LAC/output_labels'
os.makedirs(labels_dir, exist_ok=True)

# Helper function to extract the prefix from file name (handles suffix like 'grayscale')
def extract_prefix(filename):
    return ''.join([c for c in filename if c.isdigit()])

# Pair grayscale and semantic images by prefix
grayscale_files = {extract_prefix(f): f for f in os.listdir(grayscale_dir) if f.endswith('grayscale.png')}
semantic_files = {extract_prefix(f): f for f in os.listdir(semantic_dir) if f.endswith('semantic.png')}

# Generate annotations
for prefix, grayscale_file in grayscale_files.items():
    if prefix not in semantic_files:
        print(f"No matching semantic mask for {grayscale_file}")
        continue

    # Read images
    grayscale_img = cv2.imread(os.path.join(grayscale_dir, grayscale_file), cv2.IMREAD_GRAYSCALE)
    semantic_mask = cv2.imread(os.path.join(semantic_dir, semantic_files[prefix]), cv2.IMREAD_GRAYSCALE)

    if grayscale_img is None or semantic_mask is None:
        print(f"Error reading images for prefix {prefix}. Skipping.")
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

            # Ensure valid bounding boxes (filter tiny boxes & edge cases)
            if 0.02 <= bbox_width <= 1.0 and 0.02 <= bbox_height <= 1.0 and 0.02 <= x_center <= 0.98 and 0.02 <= y_center <= 0.98:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

print("Annotation generation complete! Tiny boxes and edge objects filtered.")
