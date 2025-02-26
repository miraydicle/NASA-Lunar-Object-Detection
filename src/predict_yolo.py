import os
import random
import shutil
import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('runs/detect/train/weights/best.pt')

# Define the validation images folder
val_images_dir = 'C:/Users/Dell/Desktop/NASA_LAC/dataset/images/val'
output_folder = 'C:/Users/Dell/Desktop/NASA_LAC/runs/detect'  # Save directly to detect/
predict_folder = os.path.join(output_folder, 'predict')

# Ensure the validation images folder exists
if not os.path.exists(val_images_dir):
    raise FileNotFoundError("ERROR: Validation images folder not found!")

# Remove old prediction results before running new ones
if os.path.exists(predict_folder):
    shutil.rmtree(predict_folder)  # Delete previous predictions

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(val_images_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]

# Ensure there are images in the folder
if not image_files:
    raise FileNotFoundError("ERROR: No images found in the validation dataset!")

# Select 10 random images (or all if less than 10 exist)
num_images = min(10, len(image_files))
selected_images = random.sample(image_files, num_images)

print(f"Running predictions on {num_images} random images...")

# Run YOLO prediction on each selected image
for image_file in selected_images:
    image_path = os.path.join(val_images_dir, image_file)
    print(f"Processing: {image_path}")

    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Run YOLO model on the image and save results
    results = model.predict(image, save=True, project=output_folder, name="predict", exist_ok=True)

print(f"Predictions complete! Results saved in '{predict_folder}'.")
