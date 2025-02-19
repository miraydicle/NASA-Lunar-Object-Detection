from ultralytics import YOLO

# Load model
model = YOLO('yolo11n.pt')

# Move model to GPU
model.to('cuda')

# Train the model with optimized parameters
model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=416,
    workers=16,                # Workers
    batch=128,                  # Batch size
    amp=True,                # Mixed precision
    device="cuda"
)
