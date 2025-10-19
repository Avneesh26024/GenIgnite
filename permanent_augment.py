import os
import cv2
import random
import numpy as np
import shutil
from glob import glob
from tqdm import tqdm

# --- CONFIGURATION ---
ORIGINAL_TRAIN_DIR = r"train_3/train3"  # Source of your original train data
ORIGINAL_VAL_DIR = r"train_3/val3"  # Source of your original val data

OUTPUT_TRAIN_DIR = r"train4"  # Augmented train output
OUTPUT_VAL_DIR = r"val4"  # Copied val output

CROPPED_DIR = r"Least_Labelled_Classes/cropped_images"

# Augmentation settings (for training data only)
COPY_PASTE_PROB = 0.5
MIN_OBJECTS_TO_PASTE = 1
MAX_OBJECTS_TO_PASTE = 3
OBJECT_SIZE_PCT = (0.02, 0.05)


# ==========================================================
#  LOAD CROPPED OBJECTS
# ==========================================================
def load_cropped_objects(directory):
    cropped_objects = []
    if not os.path.exists(directory):
        print(f"[WARN] Cropped objects directory not found: {directory}")
        return cropped_objects

    for class_id_str in os.listdir(directory):
        class_path = os.path.join(directory, class_id_str)
        if not os.path.isdir(class_path):
            continue
        numeric_part = "".join(filter(str.isdigit, class_id_str))
        if not numeric_part:
            continue

        class_id = int(numeric_part)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            crop = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if crop is not None:
                cropped_objects.append((crop, class_id))
    return cropped_objects


# ==========================================================
#  COPY-PASTE AUGMENTATION
# ==========================================================
def paste_object(bg_img, bg_labels, crop_img, class_id):
    if crop_img is None:
        return bg_img, bg_labels

    if len(bg_img.shape) == 2:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)
    if len(crop_img.shape) == 2:
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)

    bg_h, bg_w, _ = bg_img.shape
    has_alpha = (len(crop_img.shape) == 3 and crop_img.shape[2] == 4)
    crop_h, crop_w = crop_img.shape[:2]

    area_frac = random.uniform(*OBJECT_SIZE_PCT)
    try:
        scale_factor = np.sqrt((bg_h * bg_w * area_frac) / (crop_w * crop_h))
    except ZeroDivisionError:
        return bg_img, bg_labels

    new_w, new_h = max(1, int(crop_w * scale_factor)), max(1, int(crop_h * scale_factor))
    if new_w >= bg_w or new_h >= bg_h:
        return bg_img, bg_labels

    crop_resized = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_offset, y_offset = random.randint(0, bg_w - new_w), random.randint(0, bg_h - new_h)

    if has_alpha:
        crop_f = crop_resized.astype(np.float32) / 255.0
        bg_f = bg_img.astype(np.float32) / 255.0
        alpha = crop_f[:, :, 3:4]
        crop_rgb = crop_f[:, :, :3]
        roi = bg_f[y_offset:y_offset + new_h, x_offset:x_offset + new_w, :3]
        blended = alpha * crop_rgb + (1 - alpha) * roi
        bg_f[y_offset:y_offset + new_h, x_offset:x_offset + new_w, :3] = blended
        bg_img = (bg_f * 255).astype(np.uint8)
    else:
        bg_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = crop_resized

    # YOLO-format normalized bbox
    x_center = (x_offset + new_w / 2) / bg_w
    y_center = (y_offset + new_h / 2) / bg_h
    w_norm, h_norm = new_w / bg_w, new_h / bg_h
    bg_labels.append(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}")
    return bg_img, bg_labels


# ==========================================================
#  AUGMENT TRAINING DATASET
# ==========================================================
def create_augmented_dataset(orig_dir, out_dir, cropped_objects):
    print(f"\n[INFO] Starting TRAIN dataset creation...")
    print(f"  Original data: {orig_dir}")
    print(f"  Output data:   {out_dir}")

    orig_img_dir = os.path.join(orig_dir, "images")
    orig_lbl_dir = os.path.join(orig_dir, "labels")

    out_img_dir = os.path.join(out_dir, "images")
    out_lbl_dir = os.path.join(out_dir, "labels")

    if os.path.exists(out_dir):
        print(f"[WARN] Output directory {out_dir} already exists. Removing it.")
        shutil.rmtree(out_dir)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    print(f"[INFO] Created new directory: {out_dir}")

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob(os.path.join(orig_img_dir, ext)))

    if not image_paths:
        print(f"[ERROR] No images found in {orig_img_dir}. Exiting.")
        return

    print(f"[INFO] Found {len(image_paths)} train images to process.")

    augmented_count = 0
    for img_path in tqdm(image_paths, desc="Augmenting TRAIN set"):
        img_filename = os.path.basename(img_path)
        name_part, _ = os.path.splitext(img_filename)

        lbl_path = os.path.join(orig_lbl_dir, f"{name_part}.txt")

        out_img_path = os.path.join(out_img_dir, img_filename)
        out_lbl_path = os.path.join(out_lbl_dir, f"{name_part}.txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read {img_path}, skipping.")
            continue

        labels_list = []
        if os.path.exists(lbl_path):
            try:
                with open(lbl_path, 'r') as f:
                    labels_list = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"[WARN] Could not read label {lbl_path}: {e}, skipping.")
                continue

        if random.random() < COPY_PASTE_PROB and cropped_objects:
            augmented_count += 1
            num_to_paste = random.randint(MIN_OBJECTS_TO_PASTE, MAX_OBJECTS_TO_PASTE)
            augmented_img = img.copy()

            for _ in range(num_to_paste):
                try:
                    crop_img, class_id = random.choice(cropped_objects)
                    augmented_img, labels_list = paste_object(
                        augmented_img, labels_list, crop_img, class_id
                    )
                except Exception as e:
                    print(f"[WARN] Failed to paste object on {img_filename}: {e}")

            cv2.imwrite(out_img_path, augmented_img)

        else:
            shutil.copyfile(img_path, out_img_path)

        if labels_list:
            try:
                with open(out_lbl_path, 'w') as f:
                    f.write("\n".join(labels_list))
            except Exception as e:
                print(f"[WARN] Failed to write label {out_lbl_path}: {e}")
        else:
            open(out_lbl_path, 'w').close()

    print(f"\n[INFO] --- TRAIN Set Complete ---")
    print(f"  Total images: {len(image_paths)}, Augmented: {augmented_count}")
    print(f"  Train dataset ready at: {out_dir}")


# ==========================================================
#  COPY VALIDATION DATASET
# ==========================================================
def copy_validation_dataset(orig_dir, out_dir):
    print(f"\n[INFO] Starting VAL dataset creation...")
    print(f"  Original data: {orig_dir}")
    print(f"  Output data:   {out_dir}")

    if not os.path.exists(orig_dir):
        print(f"[ERROR] Original validation directory not found: {orig_dir}")
        return

    if os.path.exists(out_dir):
        print(f"[WARN] Output directory {out_dir} already exists. Removing it.")
        shutil.rmtree(out_dir)

    try:
        # Simply copy the entire folder
        shutil.copytree(orig_dir, out_dir)
        print(f"[INFO] --- VAL Set Complete ---")
        print(f"  Successfully copied validation set to: {out_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to copy validation set: {e}")


# ==========================================================
#  MAIN
# ==========================================================
if __name__ == '__main__':
    print("[INFO] Loading cropped objects for augmentation...")
    loaded_cropped_objects = load_cropped_objects(CROPPED_DIR)
    print(f"[INFO] Loaded {len(loaded_cropped_objects)} cropped objects.")

    if not loaded_cropped_objects:
        print("[ERROR] No cropped objects loaded. Cannot perform augmentation.")
    else:
        # 1. Create augmented training set
        create_augmented_dataset(
            orig_dir=ORIGINAL_TRAIN_DIR,
            out_dir=OUTPUT_TRAIN_DIR,
            cropped_objects=loaded_cropped_objects
        )

    # 2. Create a clean copy of the validation set
    copy_validation_dataset(
        orig_dir=ORIGINAL_VAL_DIR,
        out_dir=OUTPUT_VAL_DIR
    )

    print("\n[INFO] Dataset creation process finished.")