import yaml
from pathlib import Path
from ultralytics import YOLO


def evaluate_model(model_path, data_config_path):
    """
    Loads a trained YOLOv8 model and evaluates its performance on the test set,
    displaying key metrics like mAP, Precision, and Recall.

    Args:
        model_path (str or Path): The path to the trained .pt model file.
        data_config_path (str or Path): The path to the yolo_params.yaml file.
    """
    # --- 1. Input Validation ---
    model_file = Path(model_path)
    data_file = Path(data_config_path)

    if not model_file.exists():
        print(f"Error: Model file not found at '{model_file}'")
        return

    if not data_file.exists():
        print(f"Error: Data configuration file not found at '{data_file}'")
        print("Please make sure 'yolo_params.yaml' is in the same folder as this script.")
        return

    # Check if the yaml file specifies a test set
    try:
        with open(data_file, 'r') as f:
            data_config = yaml.safe_load(f)
            if 'test' not in data_config or not data_config['test']:
                print(f"Error: Your data config file ('{data_file}') must specify a 'test' dataset path.")
                return
    except (yaml.YAMLError, FileNotFoundError) as e:
        print(f"Error reading or parsing the YAML file: {e}")
        return


    # --- 2. Load the Model ---
    print(f"Loading model from '{model_file}'...")
    try:
        model = YOLO(model_file)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 3. Run Validation ---
    print(f"\nEvaluating model on the test set specified in '{data_file}'...")
    print("-" * 70)

    try:
        # The model.val() function will automatically find the test set using the 'split' argument
        # and print all the final scores and metrics to the console.
        metrics = model.val(data=str(data_file), split="test", conf=0.25, iou=0.5)

        print("-" * 70)
        print("Evaluation complete. Here is a summary of the key metrics:")

        # --- 4. Display Key Metrics ---
        # mAP50-95 (Mean Average Precision at IoU thresholds from 0.5 to 0.95)
        # This is the primary metric for object detection challenges like COCO.
        map50_95 = metrics.box.map
        print(f"  - mAP@.50-.95 (Primary Metric): {map50_95:.4f}")

        # mAP50 (Mean Average Precision at a single IoU threshold of 0.5)
        # This metric is a good indicator of the model's ability to correctly locate objects.
        map50 = metrics.box.map50
        print(f"  - mAP@.50: {map50:.4f}")

        # Precision: "Of all the boxes the model predicted, how many were correct?"
        # A high precision means the model has few false positives.
        precision = metrics.box.mp
        if precision is not None:
             print(f"  - Mean Precision: {precision:.4f}")


        # Recall: "Of all the actual objects in the images, how many did the model find?"
        # A high recall means the model has few false negatives.
        recall = metrics.box.mr
        if recall is not None:
            print(f"  - Mean Recall: {recall:.4f}")

        print("\nNote on 'Accuracy': For object detection, we don't use a single 'accuracy' score.")
        print("Instead, mAP, Precision, and Recall give a much more complete picture of performance.")

    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")


if __name__ == '__main__':
    # --- Define Paths Directly in the Code ---

    # 1. Path to your trained model file.
    #    (Using a raw string r"..." is good practice for Windows paths)
    MODEL_TO_EVALUATE = r"C:\Users\Avneesh\Desktop\YOLO_FINETUNE\runs\detect\train15\weights\best.pt"

    # 2. Path to your dataset configuration file.
    #    This should be the YAML file that defines your dataset paths and class names.
    DATA_CONFIG_FILE = 'yolo_params.yaml'

    # --- Run the Evaluation Function ---
    evaluate_model(MODEL_TO_EVALUATE, DATA_CONFIG_FILE)
