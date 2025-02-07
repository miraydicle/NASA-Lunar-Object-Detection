
# **Lunar Surface Object Detection with YOLO**

## **Project Overview**
This project is for YOLO training of lunar simulation images for object detection purposes. The system aims to explore and map the surface of the moon through object detection and visual-inertial SLAM.

---

## **Steps for Training**

1. **Create Annotations:**
    - Create an `output_labels` folder on your desktop.
    - Run the `generate_annotations.py` script to generate annotations from the images.
    - This script uses semantic images to create YOLO-compatible labels, which will appear in the `output_labels` folder.
    - Make sure your folder paths and code match.

2. **Split Dataset:**
    - Run the `split_dataset.py` script to divide the dataset into training and validation sets.
    
3. **Start YOLO Training:**
    - Run the `train_yolo.py` script to start training.
    - The training process might take significant time, depending on your computer's GPU.

4. **Run Inference:**
    - Once training is successful, run the `predict_yolo.py` script to perform object detection on test images.

    **GOOD LUCK!**

---

## **Features**
- **Object Detection** using Ultralytics YOLO, built on **PyTorch**.
- **Dataset Generation:** Prepares images and labels from grayscale and semantic input images.
- **Training Pipeline:** Supports optimized YOLO model training with image augmentations, batch processing, and mixed precision.

---

## **Project Structure**

```
project-directory/
│
├── training_images/                # Input images (grayscale and semantic)
│   ├── grayscale/                  # Grayscale images for training
│   └── semantic/                   # Semantic segmentation masks
│
├── output_labels/                  # Generated YOLO-compatible annotation files
│
├── dataset/                        # Final dataset structure
│   ├── images/                     # Training and validation images
│   │   ├── train/
│   │   └── val/
│   └── labels/                     # YOLO annotation files
│       ├── train/
│       └── val/
│
├── scripts/                        # Project scripts
│   ├── generate_annotations.py     # Script to generate YOLO annotations
│   ├── split_dataset.py            # Script to split images into train/val sets
│   └── train_yolo.py               # Script to train the YOLO model
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd project-directory
```

### **2. Create and Activate an Anaconda Environment**
```bash
conda create -n moon-detection python=3.9 -y
conda activate moon-detection
```

### **3. Install Dependencies**
Install the required packages:
```bash
pip install -r requirements.txt
```

Alternatively, if using Anaconda:
```bash
conda env update --file environment.yml
```

### **4. Generate Annotations**
Run the script to generate YOLO-compatible annotations from input images:
```bash
python scripts/generate_annotations.py
```

### **5. Split Dataset**
Organize the dataset into training and validation sets:
```bash
python scripts/split_dataset.py
```

### **6. Train the YOLO Model**
Train the YOLO model with optimized settings:
```bash
python scripts/train_yolo.py
```

---

## **Training Optimizations**
- **Mixed Precision:** Enabled by default (`amp=True`).
- **Batch Size:** Adjust based on your system's resources.
- **Image Size:** Default is `416x416` for faster training. Modify in the script if necessary.

