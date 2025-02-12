from ultralytics import YOLO

# Load model
model = YOLO('yolo11n.pt')

# Train the model with optimized parameters
model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=416,
    workers=4,                # Workers
    batch=4,                  # Batch size
    amp=False,                # Mixed precision
    device='cpu'              # Force training on CPU if needed
)
