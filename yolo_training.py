from ultralytics import YOLO
import torch # To check for GPU
import mlflow
import os

# Define Hyperparameters
PRETRAINED_MODEL_NAME = 'yolov8n.pt' # or 'yolov8s.pt', 'yolov8m.pt'
DATA_YAML = 'dataset/data.yaml' # Path from Step 2.3
EPOCHS = 50  # Start with a moderate number, e.g., 25-100
IMAGE_SIZE = 640 # Input image size for the model
BATCH_SIZE = 2   # Adjust based on your GPU memory (e.g., 4, 8, 16)
PROJECT_NAME = 'YOLOv8_KU_PARKING'
RUN_NAME = 'exp_ku_parking_yolov8n_50epochs_2_batch_2'

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("YOLOv8_KU_PARKING")

# Check for GPU
if torch.backends.mps.is_available():
    print("MPS is available! PyTorch can use the GPU on Apple Silicon.")
    device = torch.device("mps")
else:
    print("MPS not available. PyTorch will use CPU.")
    device = torch.device("cpu")
print(f"Using device: {device}")

# Start an MLflow run
with mlflow.start_run(run_name=RUN_NAME) as run:
    print(f"MLflow Run ID: {run.info.run_id}")
    mlflow.log_param("pretrained_model", PRETRAINED_MODEL_NAME)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("image_size", IMAGE_SIZE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("data_yaml", DATA_YAML)
    mlflow.log_param("device", device)

    # Load a pre-trained YOLO model
    model = YOLO(PRETRAINED_MODEL_NAME)

    # Train the model
    # Ultralytics automatically logs metrics, checkpoints, and some artifacts to MLflow
    # if mlflow is imported and an experiment is active.
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME, # All runs will be saved under 'yolo_ku_parking_runs/'
        name=RUN_NAME,        # This specific run: 'yolo_ku_parking_runs/exp_ku_parking_yolov8n_50epochs'
        device=device,        # Use '0' for first GPU, or 'cpu'
        exist_ok=True,        # Overwrite existing run if name conflicts
    )

    # After training, the best model is usually saved as:
    # 'yolo_ku_parking_runs/exp_ku_parking_yolov8n_50epochs/weights/best.pt'
    best_model_path = os.path.join(PROJECT_NAME, RUN_NAME, 'weights', 'best.pt')

    if os.path.exists(best_model_path):
        print(f"Best model saved at: {best_model_path}")
        mlflow.log_artifact(best_model_path, artifact_path="yolo_model_weights")

        run_output_dir = os.path.join(PROJECT_NAME, RUN_NAME)
        for item in ['results.csv', 'confusion_matrix.png', 'F1_curve.png', 'PR_curve.png']:
            item_path = os.path.join(run_output_dir, item)
            if os.path.exists(item_path):
                mlflow.log_artifact(item_path, artifact_path="training_results")
            else:
                print(f"Warning: {item_path} not found for logging.")
    else:
        print(f"Error: Best model not found at {best_model_path}")

    print("Training finished. Check MLflow UI for metrics and artifacts.")
