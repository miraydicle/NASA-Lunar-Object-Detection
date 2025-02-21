from ultralytics import YOLO
import torch

# Load YOLO model
model = YOLO('yolo11n.pt')  # Ensure correct path
model.to('cuda')

# Optimized Training Parameters for Faster Training & Better Predictions
model.train(
    data='dataset.yaml',    # Path to dataset configuration
    epochs=100,             # Reduced to avoid unnecessary long training
    imgsz=512,              # Reduced image size for speed & efficiency
    workers=16,             # More workers for faster data loading
    batch=128,         # Increased batch size for faster training
    optimizer='AdamW',      # More adaptive optimizer
    lr0=0.001,              # Fine-tuned learning rate
    lrf=0.01,               # Final learning rate factor
    momentum=0.937,         # Momentum for stability
    weight_decay=0.0005,    # Regularization to prevent overfitting
    amp=True,               # Mixed precision training
    device="cuda",          # Use CUDA if available
    val=True,               # Enable validation tracking
    save_period=10,         # Save model every 10 epochs
    patience=20,            # Stop early if no improvement
    hsv_h=0.015,            # Hue augmentation
    hsv_s=0.7,              # Saturation augmentation
    hsv_v=0.4,              # Value (brightness) augmentation
    flipud=0.5,             # Random vertical flipping
    fliplr=0.5,             # Random horizontal flipping
    mosaic=1.0,             # Enable Mosaic augmentation for first 50 epochs
    mixup=0.2,              # Enable MixUp augmentation for first 50 epochs
    box=0.05,               # Higher IoU threshold for better predictions
    cls=0.5,                # Improved class confidence tuning
    conf=0.25,              # Confidence threshold tuning
)

print("Training complete! Best model saved with optimized speed and prediction accuracy.")
