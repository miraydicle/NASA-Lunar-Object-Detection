from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('runs/train/exp/weights/best.pt')

# Read and test an image
image = cv2.imread('path/to/test_image.png')
results = model.predict(image)

# Display the results
results.show()
