from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Read and test an image
image_path = 'C:/Users/Dell/Desktop/dataset/images/val/2grayscale.png'
image = cv2.imread(image_path)

# Run YOLO prediction
results = model.predict(image, save=True, project="runs/detect", name="predict")

# Display results
for i, result in enumerate(results):
    result.show()  # Show the detection result

print("Prediction complete!")
