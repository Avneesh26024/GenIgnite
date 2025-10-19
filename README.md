# 🧠 Improving YOLO Object Detection via Targeted Copy-Paste Augmentation

This repository showcases our approach to enhancing object detection performance on an imbalanced dataset using a targeted copy-paste augmentation strategy. We identified underrepresented classes, augmented them intelligently, and evaluated performance across multiple YOLO variants to analyze the trade-offs between model size and accuracy.

---

## ⚠️ Important Note on Paths
Before running any scripts, you **must update the file paths** inside each Python script (`crop_images.py`, `permanent_augment.py`, `train.py`) to match your local directory structure.  

**Files that require path changes:**
- `crop_images.py`: `images_dir`, `labels_dir`, `output_dir`  
- `permanent_augment.py`: `ORIGINAL_TRAIN_DIR`, `ORIGINAL_VAL_DIR`, `OUTPUT_TRAIN_DIR`, `OUTPUT_VAL_DIR`, `CROPPED_DIR`  
- `train.py`: `data` argument in `model.train()` YAML file path  

Make sure these paths point to the correct locations on your machine before running the pipeline.

---

## 🚀 Approach Overview

### 1. Identifying Underrepresented Classes
We analyzed the dataset distribution and found certain object classes that were significantly underrepresented. These classes tended to have lower recall and mAP scores, negatively affecting the model’s overall performance.

### 2. Targeted Copy-Paste Augmentation
To address this imbalance, we cropped images of objects belonging to these underrepresented classes and applied a probabilistic copy-paste augmentation technique. These cropped objects were randomly resized and pasted onto other training images, creating new, diverse samples while retaining realistic context.

**Augmentation Highlights:**
- Random chance (50%) of applying augmentation per image  
- 1–3 objects pasted per image  
- Random object scaling between 2%–5% of total image area  
- Automatic YOLO-format bounding box generation  

This augmentation enriched the training data without requiring new labeled samples.

---

## ⚙️ Relevant Files

| File | Purpose |
|------|---------|
| `crop_images.py` | Crops and saves images of underrepresented classes. |
| `permanent_augment.py` | Creates an augmented dataset by randomly pasting cropped images onto base training images. |
| `train.py` | Fine-tunes YOLO models (YOLOv8s, YOLOv8m, YOLO11n) using the augmented dataset. |
| `yolo_params2.yaml` | Defines train/val/test datasets, class names, and number of classes. |

---

## 🧩 Reproducing Our Results

### Step 0️⃣ – Set Paths
1. Update paths in all scripts (`crop_images.py`, `permanent_augment.py`, `train.py`) to match your folders.  
2. Update the dataset paths in `yolo_params2.yaml` to point to your `train`, `val`, and `test` images.

### Step 1️⃣ – Crop Least Represented Classes
```bash
python crop_images.py
```
This extracts and stores cropped instances of the least represented classes in a separate folder.

### Step 2️⃣ – Create Augmented Dataset
```bash
python permanent_augment.py
```
This script uses the cropped objects to generate a new augmented dataset, enriching the underrepresented categories.

### Step 3️⃣ – Train the Model
```bash
python train.py
```
Fine-tunes the YOLO model using the augmented dataset. You can switch between YOLO versions (YOLOv8s, YOLOv8m, YOLO11n) to compare results.

---

## 📊 Experimental Results

### 🔹 YOLOv8s — 11M Parameters
| Metric | Score |
|--------|-------|
| mAP@0.5 | 0.768 |
| mAP@0.5:0.95 | 0.671 |
| Precision | 0.884 |
| Recall | 0.600 |

### 🔹 YOLO11n — 2.6M Parameters
| Metric | Score |
|--------|-------|
| mAP@0.5 | 0.700 |
| mAP@0.5:0.95 | 0.597 |
| Precision | 0.837 |
| Recall | 0.512 |

### 🔹 YOLOv8m — 25.8M Parameters
| Metric | Score |
|--------|-------|
| mAP@0.5 | 0.803 |
| mAP@0.5:0.95 | 0.722 |
| Precision | 0.936 |
| Recall | 0.637 |

---

## 🧠 Key Takeaways
- Dataset balancing via copy-paste augmentation led to measurable improvements in both mAP and recall.  
- YOLOv8m achieved the highest accuracy, while YOLOv8s offered a strong trade-off between performance and model size.  
- The augmentation strategy provided data diversity without requiring any new labeled samples.

---

## ⚙️ Environment Details

| Component | Version / Hardware |
|-----------|------------------|
| Python | 3.10 |
| PyTorch | 2.5.1 |
| Ultralytics | 8.3.217 |
| GPU | NVIDIA GeForce RTX 3070 Ti (8 GB VRAM) |
| CUDA | Enabled |

---

## 📁 Model and Results Locations

| Model | Training Folder | Evaluation Folder |
|-------|----------------|-----------------|
| YOLOv8m | `runs/detect/train14/` | `runs/detect/val3/` |
| YOLOv8s | `runs/detect/train13/` | `runs/detect/val4/` |
| YOLO11n | `runs/detect/train15/` | `runs/detect/val5/` |

Each folder contains the `best.pt` weights and evaluation logs.

---

## 🧾 Summary Table

| Model | Parameters (M) | mAP@0.5 | mAP@0.5:.95 | Precision | Recall |
|-------|----------------|---------|--------------|-----------|--------|
| YOLO11n | 2.6 | 0.700 | 0.597 | 0.837 | 0.512 |
| YOLOv8s | 11.1 | 0.768 | 0.671 | 0.884 | 0.600 |
| YOLOv8m | 25.8 | 0.803 | 0.722 | 0.936 | 0.637 |

---

## 🏁 Conclusion

Through targeted augmentation focused on rare classes, we achieved a notable boost in detection performance across all YOLO variants. This demonstrates that strategic data enrichment can be as impactful as model architecture improvements in real-world object detection tasks.

---
