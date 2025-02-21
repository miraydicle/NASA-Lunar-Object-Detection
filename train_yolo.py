from ultralytics import YOLO
import torch

# Ensure CUDA is available and use it
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Load model
model = YOLO('yolo11n.pt')
model.to(device)

# Optimized Training Parameters with Data Augmentation
model.train(
    data='dataset.yaml',    # Path to dataset configuration
    epochs=150,             # Increased for better convergence
    imgsz=640,              # Higher resolution for better object detection
    workers=8,              # Balanced number of workers for stability
    batch_size=64,          # Adjusted for GPU memory efficiency
    optimizer='AdamW',      # More adaptive optimizer
    lr0=0.001,              # Fine-tuned learning rate
    lrf=0.01,               # Final learning rate factor
    momentum=0.937,         # Momentum for stability
    weight_decay=0.0005,    # Regularization to prevent overfitting
    amp=True,               # Mixed precision training
    device=device,          # Use CUDA if available
    val=True,               # Enable validation tracking
    save_period=10,         # Save model every 10 epochs
    patience=20,            # Stop early if no improvement
    hsv_h=0.015,            # Hue augmentation
    hsv_s=0.7,              # Saturation augmentation
    hsv_v=0.4,              # Value (brightness) augmentation
    flipud=0.5,             # Random vertical flipping
    fliplr=0.5,             # Random horizontal flipping
    mosaic=1.0,             # Enable Mosaic augmentation
    mixup=0.2,              # Enable MixUp augmentation
)

print("Training complete! Best model saved.")
