import os
import cv2

# -----------------------------
# CONFIGURATION
# -----------------------------
images_dir = r"C:\Users\Avneesh\Desktop\Genignite\train_3\train3\images"
labels_dir = r"C:\Users\Avneesh\Desktop\Genignite\train_3\train3\labels"
output_dir = r"C:\Users\Avneesh\Desktop\Genignite\Hackathon2_scripts\Lease_Labelled_Classes\cropped_images"  # folder to save cropped objects

target_classes = [3, 4, 5]  # classes to crop

# Create output directories
for cls in target_classes:
    os.makedirs(os.path.join(output_dir, f"class{cls}"), exist_ok=True)

# Counter for naming cropped images
counters = {cls: 0 for cls in target_classes}

# -----------------------------
# FUNCTION TO CROP OBJECTS
# -----------------------------
def yolo_to_bbox(x_center, y_center, w, h, img_w, img_h):
    """Convert normalized YOLO format to pixel coordinates"""
    x1 = int((x_center - w / 2) * img_w)
    y1 = int((y_center - h / 2) * img_h)
    x2 = int((x_center + w / 2) * img_w)
    y2 = int((y_center + h / 2) * img_h)
    # Ensure coordinates are within image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
    return x1, y1, x2, y2

# -----------------------------
# PROCESS ALL IMAGES
# -----------------------------
for img_file in os.listdir(images_dir):
    if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(images_dir, img_file)
    label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")

    # Skip if label file does not exist
    if not os.path.exists(label_path):
        continue

    # Read image
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    # Read labels
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.strip() == "":
            continue
        parts = line.strip().split()
        class_id = int(parts[0])

        if class_id in target_classes:
            x_c, y_c, w, h = map(float, parts[1:5])
            x1, y1, x2, y2 = yolo_to_bbox(x_c, y_c, w, h, img_w, img_h)
            crop = img[y1:y2, x1:x2]

            # Skip empty crops
            if crop.size == 0:
                continue

            # Update counter and save
            counters[class_id] += 1
            crop_name = f"class{class_id}_{counters[class_id]:04d}.jpg"
            crop_path = os.path.join(output_dir, f"class{class_id}", crop_name)
            cv2.imwrite(crop_path, crop)

            # Save YOLO label for cropped image (object fills entire image)
            label_name = os.path.splitext(crop_name)[0] + ".txt"
            label_path_out = os.path.join(output_dir, f"class{class_id}", label_name)
            with open(label_path_out, "w") as lf:
                lf.write(f"{class_id} 0.5 0.5 1.0 1.0\n")  # object covers full crop

print("✅ Cropping complete!")
for cls in target_classes:
    print(f"Class {cls}: {counters[cls]} objects cropped.")
