from ultralytics import YOLO
import torch

# Load YOLO model
model = YOLO('yolo11n.pt')  # Ensure correct path
model.to('cuda')

# Optimized Training Parameters for Faster Training & Better Predictions
model.train(
    data='dataset.yaml',    # Path to dataset configuration
    epochs=120,             # Reduced to avoid unnecessary long training
    imgsz=640,              # Reduced image size for speed & efficiency
    workers=16,             # More workers for faster data loading
    batch=64,               # Increased batch size for faster training
    optimizer='AdamW',      # More adaptive optimizer
    lr0=0.0008,             # Lower learning rate for fine-tuning precision
    lrf=0.01,               # Final learning rate factor
    momentum=0.937,         # Momentum for stability
    weight_decay=0.0005,    # Regularization to prevent overfitting
    amp=True,               # Mixed precision training
    device="cuda",          # Use CUDA if available
    val=True,               # Enable validation tracking
    save_period=10,         # Save model every 10 epochs
    patience=20,            # Stop early if no improvement
    augment=True,           # Advanced YOLO augmentation enabled
    conf=0.6,               # Confidence threshold tuning
    iou=0.45                # Enable Non-Maximum Suppression (NMS) to remove overlapping boxes
)

print("Training complete! Best model saved with optimized speed and prediction accuracy.")
