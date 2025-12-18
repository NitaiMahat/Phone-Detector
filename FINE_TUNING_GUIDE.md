# YOLOv5 Fine-Tuning Guide for Phone Detection

## Overview

This guide will help you fine-tune YOLOv5 Nano specifically for phone detection in your Focus Guard project.

---

## Step 1: Set Up Environment

### Prerequisites
- Python 3.8 or higher
- GPU recommended (NVIDIA with CUDA) but CPU works too
- 8GB+ RAM

### Install YOLOv5

```bash
# Create a new folder for training (outside your web project)
mkdir phone-detection-training
cd phone-detection-training

# Clone YOLOv5
git clone https://github.com/ultralytics/yolov5
cd yolov5

# Install dependencies
pip install -r requirements.txt

# Verify installation
python detect.py --weights yolov5n.pt --source 0  # Test with webcam
```

---

## Step 2: Create Your Dataset

### Option A: Use Roboflow (Easiest - Recommended)

1. Go to [Roboflow](https://roboflow.com/) and create a free account
2. Create a new project → Object Detection
3. Upload images of phones in various scenarios
4. Label phones with bounding boxes
5. Export in "YOLOv5 PyTorch" format
6. Download and extract to `yolov5/datasets/phones/`

### Option B: Collect Images Manually

#### Folder Structure
```
phone-detection-training/
└── yolov5/
    └── datasets/
        └── phones/
            ├── images/
            │   ├── train/
            │   │   ├── img001.jpg
            │   │   ├── img002.jpg
            │   │   └── ...
            │   └── val/
            │       ├── img100.jpg
            │       └── ...
            └── labels/
                ├── train/
                │   ├── img001.txt
                │   ├── img002.txt
                │   └── ...
                └── val/
                    ├── img100.txt
                    └── ...
```

#### What Images to Collect (500-1000 minimum)

**Must-have scenarios:**
- [ ] Phone in hand (front view)
- [ ] Phone in hand (side view)
- [ ] Phone in hand (back of phone visible)
- [ ] Phone on desk/table
- [ ] Phone partially visible (in pocket)
- [ ] Phone at different distances (close, medium, far)
- [ ] Different lighting (bright, dim, natural, artificial)
- [ ] Different phone models (iPhone, Android, various sizes)
- [ ] Different backgrounds (office, home, classroom)
- [ ] Different skin tones holding phones

**How to collect:**
1. Use your webcam to take screenshots
2. Download from Google Images (search "person holding phone")
3. Use phone detection datasets from Kaggle
4. Record yourself and extract frames

#### Quick Image Capture Script

Save this as `capture_images.py` in your yolov5 folder:

```python
import cv2
import os
from datetime import datetime

# Create directories
os.makedirs('datasets/phones/images/train', exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press 'c' to capture, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Capture', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        filename = f'datasets/phones/images/train/phone_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{count}.jpg'
        cv2.imwrite(filename, frame)
        print(f'Saved: {filename}')
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f'Total images captured: {count}')
```

Run it:
```bash
python capture_images.py
```

---

## Step 3: Label Your Images

### Option A: Use LabelImg (Free, Local)

```bash
pip install labelImg
labelImg
```

1. Open Dir → Select `datasets/phones/images/train`
2. Change Save Dir → `datasets/phones/labels/train`
3. Change format to YOLO (bottom left)
4. Draw boxes around phones
5. Label as "phone" (or class 0)
6. Save (Ctrl+S)
7. Next image (D key)

### Option B: Use Roboflow (Online, Easier)

1. Upload images to Roboflow
2. Use their annotation tool
3. Draw boxes, label as "phone"
4. Export as YOLOv5 format

### Label Format (YOLO format)

Each image needs a `.txt` file with the same name:
- `img001.jpg` → `img001.txt`

Format: `class_id x_center y_center width height`
- All values normalized (0-1)
- Example: `0 0.5 0.5 0.3 0.4`

---

## Step 4: Create Dataset Configuration

Create `phones.yaml` in the `yolov5/data/` folder:

```yaml
# phones.yaml - Phone Detection Dataset

path: datasets/phones  # dataset root dir
train: images/train    # train images
val: images/val        # validation images

# Classes
names:
  0: phone
```

---

## Step 5: Train the Model

### Basic Training (Start Here)

```bash
cd yolov5

# Train YOLOv5 Nano (smallest, fastest)
python train.py \
    --img 640 \
    --batch 16 \
    --epochs 100 \
    --data data/phones.yaml \
    --weights yolov5n.pt \
    --name phone_detector
```

### Training Parameters Explained

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--img` | Image size | 640 (standard) |
| `--batch` | Batch size | 16 (reduce if out of memory) |
| `--epochs` | Training iterations | 100-300 |
| `--data` | Dataset config file | Your phones.yaml |
| `--weights` | Starting weights | yolov5n.pt (pre-trained) |
| `--name` | Experiment name | phone_detector |

### If You Have a GPU

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Train with GPU (faster)
python train.py \
    --img 640 \
    --batch 32 \
    --epochs 100 \
    --data data/phones.yaml \
    --weights yolov5n.pt \
    --device 0 \
    --name phone_detector
```

### If You Only Have CPU

```bash
# Train on CPU (slower but works)
python train.py \
    --img 640 \
    --batch 8 \
    --epochs 50 \
    --data data/phones.yaml \
    --weights yolov5n.pt \
    --device cpu \
    --name phone_detector
```

### Training Time Estimates

| Hardware | 500 images | 1000 images |
|----------|------------|-------------|
| GPU (RTX 3060+) | 30-60 min | 1-2 hours |
| GPU (GTX 1060) | 1-2 hours | 2-4 hours |
| CPU | 4-8 hours | 8-16 hours |

---

## Step 6: Evaluate Results

After training, check results in `runs/train/phone_detector/`:

```
runs/train/phone_detector/
├── weights/
│   ├── best.pt      # Best model (use this!)
│   └── last.pt      # Last epoch
├── results.png      # Training graphs
├── confusion_matrix.png
└── ...
```

### Test Your Model

```bash
# Test on validation images
python val.py \
    --weights runs/train/phone_detector/weights/best.pt \
    --data data/phones.yaml

# Test on webcam
python detect.py \
    --weights runs/train/phone_detector/weights/best.pt \
    --source 0

# Test on specific images
python detect.py \
    --weights runs/train/phone_detector/weights/best.pt \
    --source path/to/test/images/
```

---

## Step 7: Export to ONNX

```bash
python export.py \
    --weights runs/train/phone_detector/weights/best.pt \
    --include onnx \
    --img 640 \
    --simplify
```

This creates: `runs/train/phone_detector/weights/best.onnx`

---

## Step 8: Use in Your Project

### Copy the new model

```bash
# Copy to your web project
cp runs/train/phone_detector/weights/best.onnx /path/to/Project-Phone-Detector/yolov5n.onnx
```

### Update detection-engine.js

Since you trained with only 1 class (phone), update the class index:

```javascript
// Change from COCO class 67 to your custom class 0
const PHONE_CLASS_INDEX = 0;  // Was 67

// Update CLASS_NAMES
const CLASS_NAMES = ['phone'];  // Only one class now
```

### Update postprocess function

Change the class filtering:

```javascript
// In postprocess function, change:
if (maxClass !== 67) continue;  // OLD (COCO)

// To:
if (maxClass !== 0) continue;   // NEW (your custom model)
```

---

## Quick Reference Commands

```bash
# 1. Setup
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

# 2. Capture training images
python capture_images.py

# 3. Label images
pip install labelImg
labelImg

# 4. Train
python train.py --img 640 --batch 16 --epochs 100 --data data/phones.yaml --weights yolov5n.pt --name phone_detector

# 5. Test
python detect.py --weights runs/train/phone_detector/weights/best.pt --source 0

# 6. Export
python export.py --weights runs/train/phone_detector/weights/best.pt --include onnx --simplify

# 7. Copy to project
cp runs/train/phone_detector/weights/best.onnx ../Project-Phone-Detector/yolov5n.onnx
```

---

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size: `--batch 8` or `--batch 4`

### "No module named 'torch'"
- Run: `pip install torch torchvision`

### Poor detection accuracy
- Add more training images (aim for 1000+)
- Train for more epochs (200-300)
- Include more diverse scenarios

### Model too slow in browser
- Use yolov5n (nano) not larger models
- Keep image size at 640

---

## Dataset Sources

### Free Phone Datasets

1. **Roboflow Universe**: https://universe.roboflow.com/
   - Search "phone detection" or "cell phone"
   - Many pre-labeled datasets available

2. **Kaggle**: https://www.kaggle.com/
   - Search "phone detection dataset"

3. **Open Images Dataset**: https://storage.googleapis.com/openimages/web/index.html
   - Has "Mobile phone" category

### Augmentation (Increase Dataset Size)

Roboflow can automatically augment your images:
- Rotation
- Brightness changes
- Blur
- Noise
- Flip

This can turn 500 images into 2000+ variations!

---

## Expected Results

After fine-tuning with 500-1000 images:

| Metric | Pre-trained | Fine-tuned |
|--------|-------------|------------|
| Accuracy | 70-80% | 90-95% |
| False Positives | Common | Rare |
| Partial Phones | Misses | Detects |
| Your Lighting | Variable | Consistent |

---

## Next Steps

1. [ ] Set up YOLOv5 environment
2. [ ] Collect 500+ images
3. [ ] Label all images
4. [ ] Create phones.yaml config
5. [ ] Train model (start with 50 epochs)
6. [ ] Evaluate results
7. [ ] Export to ONNX
8. [ ] Update detection-engine.js
9. [ ] Test in browser

Good luck! 🎯

