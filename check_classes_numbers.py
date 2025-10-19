import os
import cv2
import random
import numpy as np
import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER

# =================================================================================
# CONFIGURATION
# =================================================================================

# --- Directory containing the cropped images of least represented classes.
# --- The structure should be:
# --- cropped_dir/
# ---    class_id_0/
# ---        image1.png
# ---        image2.png
# ---    class_id_1/
# ---        image3.png
# --- ... and so on.
# --- `class_id_0` should be the integer index of the class.
cropped_dir = r"Least_Labelled_Classes/cropped_images"

# --- Probability of applying the copy-paste augmentation to a given training image.
copy_paste_prob = 0.75  # Increased for more significant impact

# --- Range for the size of the pasted object, as a percentage of the background image area.
object_size_pct = (0.05, 0.15)


# =================================================================================
# UTILITY FUNCTIONS
# =================================================================================

def load_cropped_objects(directory):
    """
    Loads cropped object images from a specified directory structure.

    Args:
        directory (str): The path to the directory containing class subfolders.

    Returns:
        list: A list of tuples, where each tuple contains an image (as a NumPy array)
              and its corresponding class ID (int).
    """
    cropped_objects = []
    if not os.path.exists(directory):
        LOGGER.warning(f"Cropped objects directory not found: {directory}")
        return cropped_objects

    class_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    for class_id_str in class_dirs:
        try:
            class_id = int(class_id_str)
            class_path = os.path.join(directory, class_id_str)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                # Use IMREAD_UNCHANGED to keep the alpha channel for transparency
                crop = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if crop is not None:
                    cropped_objects.append((crop, class_id))
                else:
                    LOGGER.warning(f"Could not read cropped image: {img_path}")
        except ValueError:
            LOGGER.warning(f"Skipping non-integer class directory: {class_id_str}")
        except Exception as e:
            LOGGER.error(f"Error loading cropped images from {class_id_str}: {e}")

    return cropped_objects


def paste_object(bg_img, bg_labels, crop_data, class_id):
    """
    Pastes a cropped object onto a background image and dynamically updates the labels.
    This function has been verified and remains correct.
    """
    crop_img = crop_data
    if crop_img is None:
        return bg_img, bg_labels

    bg_h, bg_w = bg_img.shape[:2]
    crop_h, crop_w = crop_img.shape[:2]

    # 1. Calculate the scale factor to resize the object
    area_frac = random.uniform(*object_size_pct)
    try:
        scale_factor = np.sqrt((bg_h * bg_w * area_frac) / (crop_w * crop_h))
    except ZeroDivisionError:
        return bg_img, bg_labels # Skip if crop has zero area

    new_w, new_h = int(crop_w * scale_factor), int(crop_h * scale_factor)

    # Skip if the resized object is too small or larger than the background
    if new_w <= 0 or new_h <= 0 or new_w >= bg_w or new_h >= bg_h:
        return bg_img, bg_labels

    crop_resized = cv2.resize(crop_img, (new_w, new_h))

    # 2. Find a random valid location to paste the object
    max_x, max_y = bg_w - new_w, bg_h - new_h
    x_offset, y_offset = random.randint(0, max_x), random.randint(0, max_y)

    # 3. Paste the object
    if crop_resized.shape[2] == 4:  # Image has an alpha channel
        alpha = crop_resized[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(0, 3):
            bg_slice = bg_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w, c]
            crop_slice = crop_resized[:, :, c]
            bg_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w, c] = \
                (alpha * crop_slice + alpha_inv * bg_slice)
    else:  # Image is BGR
        bg_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = crop_resized

    # 4. Update labels dynamically by appending the new object's coordinates
    x_center, y_center = (x_offset + new_w / 2) / bg_w, (y_offset + new_h / 2) / bg_h
    w_norm, h_norm = new_w / bg_w, new_h / bg_h
    bg_labels.append(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}")

    return bg_img, bg_labels


# =================================================================================
# 1. CUSTOM DATASET CLASS
# =================================================================================
class CustomAugmentDataset(YOLODataset):
    """
    A custom YOLO dataset that applies an intelligent copy-paste augmentation.
    """
    def __init__(self, *args, cropped_objects=None, **kwargs):
        """
        Initializes the dataset.
        Args:
            cropped_objects (list): A list of (image, class_id) tuples for augmentation.
        """
        super().__init__(*args, **kwargs)
        self.cropped_objects = cropped_objects if cropped_objects is not None else []
        LOGGER.info(f"CustomAugmentDataset initialized with {len(self.cropped_objects)} cropped objects.")


    def __getitem__(self, index):
        """
        Overrides the default __getitem__ to apply custom augmentation.
        It avoids adding a cropped object if its class is already present in the image.
        """
        data = super().__getitem__(index)

        if random.random() < copy_paste_prob and self.cropped_objects:
            img = data['img']
            labels = data.get('labels')

            # --- INTELLIGENT AUGMENTATION LOGIC ---
            # 1. Get the class IDs already present in the image.
            existing_classes = set()
            if labels is not None and len(labels) > 0:
                existing_classes = {int(l[0]) for l in labels}

            # 2. Create a pool of candidate objects to paste, excluding those whose
            #    classes are already in the image.
            paste_pool = [obj for obj in self.cropped_objects if obj[1] not in existing_classes]

            # 3. Only proceed if there are valid objects to choose from.
            if not paste_pool:
                return data # Return original data if no valid object can be pasted

            # Convert image tensor to a mutable NumPy array (BGR) for OpenCV
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Convert existing labels to a list of strings
            labels_list = []
            if labels is not None and len(labels) > 0:
                labels_list = [f"{int(l[0])} {l[1]} {l[2]} {l[3]} {l[4]}" for l in labels]

            # Select a random cropped object from the filtered pool and paste it
            crop_img, class_id = random.choice(paste_pool)
            img_aug, labels_list = paste_object(img_np, labels_list, crop_img, class_id)

            # --- Convert back to framework format ---
            img_aug_rgb = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(np.transpose(img_aug_rgb, (2, 0, 1)) / 255.0).float()

            if labels_list:
                new_labels = np.array([list(map(float, l.split())) for l in labels_list], dtype=np.float32)
            else:
                new_labels = np.zeros((0, 5), dtype=np.float32)

            # Update the data dictionary with augmented image and combined labels
            data['img'] = img_tensor
            data['labels'] = torch.from_numpy(new_labels)

        return data


# =================================================================================
# 2. CUSTOM TRAINER CLASS
# =================================================================================
class CustomTrainer(DetectionTrainer):
    """
    A custom Trainer that uses our CustomAugmentDataset.
    """
    def __init__(self, *args, cropped_objects=None, **kwargs):
        """
        Initializes the trainer.
        Args:
            cropped_objects (list): A list of (image, class_id) tuples to be passed to the dataset.
        """
        super().__init__(*args, **kwargs)
        self.cropped_objects = cropped_objects


    def build_dataset(self, img_path, mode='train', batch=16):
        """
        Overrides the dataset builder to inject our custom dataset class
        and pass the cropped objects to it.
        """
        return CustomAugmentDataset(
            img_path=img_path,
            data=self.data,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=True, # Must be True for augmentations to apply
            hyp=self.args,
            rect=self.args.rect or mode == 'val',
            cache=self.args.cache or None,
            prefix=f'{mode}: ',
            stride=self.stride,
            cropped_objects=self.cropped_objects if mode == 'train' else None # Only augment training data
        )


# =================================================================================
# MAIN EXECUTION BLOCK
# =================================================================================
if __name__ == '__main__':
    # 1. LOAD CROPPED OBJECTS
    # This is done once at the beginning of the script.
    LOGGER.info("Loading cropped objects for augmentation...")
    loaded_cropped_objects = load_cropped_objects(cropped_dir)

    if not loaded_cropped_objects:
        LOGGER.warning("WARNING: No cropped objects were loaded. Copy-paste augmentation will be disabled.")
    else:
        LOGGER.info(f"Successfully loaded {len(loaded_cropped_objects)} cropped objects.")

    # 2. DEFINE TRAINING PARAMETERS
    # These are the standard YOLOv8 training overrides.
    overrides = {
        'model': 'yolov8s.pt',
        'data': 'yolov8_params.yaml', # Ensure this path is correct
        'epochs': 100,
        'batch': 16,
        'workers': 8,
        'imgsz': 640,
        'lr0': 0.001,
        'patience': 25,
        'optimizer': 'AdamW',
        'cache': 'disk',  # Use 'disk' to save RAM, 'ram' for faster training if you have enough.
        'project': 'YOLO_Finetune_Results', # Project name for saving results
        'name': 'run_with_intelligent_aug' # Experiment name
    }

    # 3. INITIALIZE AND START TRAINING
    # We pass the loaded objects directly to our custom trainer.
    trainer = CustomTrainer(overrides=overrides, cropped_objects=loaded_cropped_objects)
    trainer.train()

    LOGGER.info("Training complete!")

