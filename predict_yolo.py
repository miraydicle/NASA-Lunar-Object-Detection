import os
import random
import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('runs/detect/train2/weights/best.pt')

# Define the validation images folder
val_images_dir = 'C:/Users/Dell/Desktop/NASA_LAC/dataset/images/val'

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

    # Run YOLO model on the image
    results = model.predict(image, save=True, project="runs/detect", name="predict")

    # Display results
    for result in results:
        result.show()  # Show the detection result

print("Predictions complete! Results saved in 'runs/detect/predict'.")
