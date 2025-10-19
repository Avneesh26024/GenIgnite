import os
import cv2
import random
import numpy as np

# =================================================================================
# CONFIGURATION
# =================================================================================
# --- Set the paths to your directories ---
# Path to the directory with background images for the demo
images_dir = r"train_3/train3/images"
# Path to the directory with cropped objects, organized in class_id subfolders
cropped_dir = r"Least_Labelled_Classes/cropped_images"

# --- Augmentation Parameters ---
# The pasted object will occupy between 5% and 15% of the background image area.
object_size_pct = (0.05, 0.15)


# =================================================================================
# UTILITY FUNCTIONS
# =================================================================================

def paste_object_and_get_label(bg_img, crop_img, class_id):
    """
    Pastes a cropped object onto a background image, handling transparency,
    and returns the augmented image along with the YOLO label string.

    Args:
        bg_img (np.array): The background image.
        crop_img (np.array): The cropped object image (can have an alpha channel).
        class_id (int): The class ID of the object being pasted.

    Returns:
        tuple: A tuple containing:
            - np.array: The augmented image.
            - str: The YOLO format label string for the pasted object.
    """
    bg_h, bg_w = bg_img.shape[:2]
    crop_h, crop_w = crop_img.shape[:2]

    # 1. Resize crop to a random fraction of the background area
    area_frac = random.uniform(*object_size_pct)
    try:
        scale_factor = np.sqrt((bg_h * bg_w * area_frac) / (crop_w * crop_h))
    except ZeroDivisionError:
        return bg_img, "" # Return original image if crop has zero area

    new_w = int(crop_w * scale_factor)
    new_h = int(crop_h * scale_factor)

    # Skip if the resized object is too small or would be larger than the background
    if new_w <= 0 or new_h <= 0 or new_w >= bg_w or new_h >= bg_h:
        return bg_img, ""

    crop_resized = cv2.resize(crop_img, (new_w, new_h))

    # 2. Determine a random position to paste the object
    max_x = bg_w - new_w
    max_y = bg_h - new_h
    x_offset = random.randint(0, max_x)
    y_offset = random.randint(0, max_y)

    # 3. Paste the object, handling transparency
    if crop_resized.shape[2] == 4:  # Check for alpha channel
        alpha = crop_resized[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(0, 3):
            bg_slice = bg_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w, c]
            crop_slice = crop_resized[:, :, c]
            bg_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w, c] = \
                (alpha * crop_slice + alpha_inv * bg_slice)
    else:  # No alpha channel
        bg_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = crop_resized

    # 4. Calculate the YOLO formatted label string
    x_center = (x_offset + new_w / 2) / bg_w
    y_center = (y_offset + new_h / 2) / bg_h
    w_norm = new_w / bg_w
    h_norm = new_h / bg_h
    label_str = f"{class_id} {x_center} {y_center} {w_norm} {h_norm}"

    return bg_img, label_str

def draw_yolo_label(image, label_str):
    """
    Draws a bounding box and class label on an image from a YOLO label string.

    Args:
        image (np.array): The image to draw on.
        label_str (str): The YOLO format label string.

    Returns:
        np.array: The image with the label drawn on it.
    """
    if not label_str:
        return image

    h, w = image.shape[:2]
    parts = label_str.split()
    class_id = int(parts[0])
    x_center, y_center, width, height = map(float, parts[1:])

    # Convert normalized YOLO coordinates to pixel coordinates
    box_w = int(width * w)
    box_h = int(height * h)
    box_x = int(x_center * w - box_w / 2)
    box_y = int(y_center * h - box_h / 2)

    # Define colors and draw the box and text
    color = (0, 255, 0) # Green
    cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2)
    label_text = f"Class ID: {class_id}"
    cv2.putText(image, label_text, (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return image


# =================================================================================
# MAIN EXECUTION BLOCK
# =================================================================================
if __name__ == '__main__':
    # 1. Pick a random background image
    try:
        bg_file = random.choice([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
        bg_path = os.path.join(images_dir, bg_file)
        bg_img = cv2.imread(bg_path)
    except (IndexError, FileNotFoundError):
        print(f"Error: Could not find or load a background image from '{images_dir}'. Please check the path.")
        exit()

    # 2. Pick a random cropped object and its class ID
    all_crops = []
    for cls_dir in os.listdir(cropped_dir):
        cls_path = os.path.join(cropped_dir, cls_dir)
        if os.path.isdir(cls_path):
            # --- FIXED LOGIC ---
            # Extract only the numbers from the directory name (e.g., "class3" -> "3")
            numeric_part = "".join(filter(str.isdigit, cls_dir))
            if numeric_part:
                try:
                    class_id = int(numeric_part)
                    for f in os.listdir(cls_path):
                        if f.lower().endswith(('.jpg', '.png')):
                            all_crops.append((os.path.join(cls_path, f), class_id))
                except ValueError:
                    # This case should now be rare
                    print(f"Could not parse class ID from directory name: {cls_dir}")
            else:
                print(f"Skipping directory with no numeric ID in name: {cls_dir}")

    if not all_crops:
        print(f"Error: Could not find any cropped images in '{cropped_dir}'. Please check the path and folder structure.")
        exit()

    crop_path, crop_class_id = random.choice(all_crops)
    # Load with IMREAD_UNCHANGED to preserve potential transparency
    crop_img = cv2.imread(crop_path, cv2.IMREAD_UNCHANGED)

    # 3. Paste the object and get the corresponding label
    # We pass a copy of the background image to keep the original clean
    augmented_img, new_label = paste_object_and_get_label(bg_img.copy(), crop_img, crop_class_id)
    print(f"Generated Label for '{os.path.basename(crop_path)}': {new_label}")

    # 4. Draw the bounding box and label on the augmented image
    img_with_box = draw_yolo_label(augmented_img, new_label)

    # 5. Display the final result
    cv2.imshow("Demo Copy-Paste Augmentation", img_with_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

