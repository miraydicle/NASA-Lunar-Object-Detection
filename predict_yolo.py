from ultralytics import YOLO
import cv2
import os
import random

# Load the trained model
model = YOLO('runs/detect/train2/weights/best.pt')

# Define the validation images folder
val_images_dir = 'C:/Users/Dell/Desktop/NASA_LAC/dataset/images/val'

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(val_images_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]

# Ensure there are images in the folder
if not image_files:
    raise FileNotFoundError("ERROR: No images found in the validation dataset!")

# Select a random image
random_image = random.choice(image_files)
image_path = os.path.join(val_images_dir, random_image)

print(f"Selected random image: {image_path}")

# Read and test the selected image
image = cv2.imread(image_path)

# Run YOLO prediction
results = model.predict(image, save=True, project="runs/detect", name="predict")

# Display results
for i, result in enumerate(results):
    result.show()  # Show the detection result

print("Prediction complete!")
