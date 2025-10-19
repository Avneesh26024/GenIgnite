import yaml
from pathlib import Path
from ultralytics import YOLO


def evaluate_model(model_path, data_config_path):
    """
    Loads a trained YOLOv8 model and evaluates its performance on the test set.

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
    with open(data_file, 'r') as f:
        data_config = yaml.safe_load(f)
        if 'test' not in data_config or not data_config['test']:
            print(f"Error: Your data config file ('{data_file}') must specify a 'test' dataset path.")
            return

    # --- 2. Load the Model ---
    print(f"Loading model from '{model_file}'...")
    try:
        model = YOLO(model_file)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 3. Run Validation ---
    print(f"Evaluating model on the test set specified in '{data_file}'...")
    print("-" * 50)

    # The model.val() function will automatically find the test set using the 'split' argument
    # and print all the final scores and metrics to the console.
    metrics = model.val(data=str(data_file), split="test", conf=0.5)

    print("-" * 50)
    print("Evaluation complete.")
    # The 'metrics' object contains all the results if you need to access them programmatically.
    map50_95 = metrics.box.map
    print(f"mAP50-95 Score: {map50_95:.4f}")


if __name__ == '__main__':
    # --- Define Paths Directly in the Code ---

    # 1. Path to your trained model file.
    #    (Using a raw string r"..." is good practice for Windows paths)
    MODEL_TO_EVALUATE = r"C:\Users\Avneesh\Desktop\best.pt"

    # 2. Path to your dataset configuration file.
    #    (This assumes it's in the same folder as this script)
    DATA_CONFIG_FILE = 'yolo_params.yaml'

    # --- Run the Evaluation Function ---
    evaluate_model(MODEL_TO_EVALUATE, DATA_CONFIG_FILE)
