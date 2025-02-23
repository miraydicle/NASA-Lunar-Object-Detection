import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load trained YOLO model
model_path = "runs/detect/train2/weights/best.pt"
model = YOLO(model_path)

# Define the folder containing validation images
val_images_dir = "C:/Users/Dell/Desktop/NASA_LAC/dataset/images/val"

# Get all validation images
image_files = [f for f in os.listdir(val_images_dir) if f.endswith(('.png', '.jpg'))]

# Select 5 test images (or fewer if there aren't enough)
num_images = min(5, len(image_files))
test_images = image_files[:num_images]  # Pick the first 5 for quick testing

print(f"Running quick test on {num_images} images...")

# Ensure detection saves outputs in a single folder
save_dir = "runs/detect/quick_test"
os.makedirs(save_dir, exist_ok=True)

# Run inference
for img_file in test_images:
    img_path = os.path.join(val_images_dir, img_file)
    print(f"Testing: {img_path}")

    results = model.predict(
        img_path,
        save=True,  # Save results
        project="runs/detect",
        name="quick_test",  # Save all results in one folder
        conf=0.3,  # Lower confidence threshold to catch small objects
        iou=0.5,  # Adjust NMS to detect multiple small objects
        imgsz=640
    )

    # Show predictions
    for result in results:
        result.show()

print("Quick test complete! Check 'runs/detect/quick_test' for results.")
