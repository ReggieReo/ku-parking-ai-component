import itertools
import os

import mlflow
import torch
from ultralytics import YOLO

# --- Configuration ---
PRETRAINED_MODEL_NAME = "yolov8n.pt" # changed to any model you like
DATA_YAML = "dataset/data.yaml"
ULTRALYTICS_OUTPUT_DIR = "yolo_car_gridsearch_outputs_v6"
IMAGE_SIZE = 640

# --- MLflow Setup ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "yolo_car_gridsearch_outputs_v6"
experiment = mlflow.set_experiment(EXPERIMENT_NAME)

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLflow Experiment Name: {experiment.name}")
print(f"MLflow Experiment ID: {experiment.experiment_id}")
print(f"MLflow Artifact URI: {experiment.artifact_location}")

# --- Hyperparameter Grid ---
param_grid = {
    "optimizer": ["auto"],
    "batch": [8],
    # "batch": [4, 8, 0.70],
    "weight_decay": [0.0005, 0.00025],
    "epochs": [50, 100],
}

keys, values = zip(*param_grid.items())
hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(
    f"\nStarting grid search with {len(hyperparameter_combinations)} combinations."
)

# --- Device Setup ---
if torch.backends.mps.is_available():
    device_to_use = "mps"
elif torch.cuda.is_available():
    device_to_use = "cuda"
else:
    device_to_use = "cpu"
print(f"Using device: {device_to_use}")

# --- Grid Search Loop ---
for i, params in enumerate(hyperparameter_combinations):
    run_name_parts = [f"grid_run_yolo_v8s_{i+1}"]
    for key, value in sorted(params.items()):
        run_name_parts.append(f"{key}-{value}")
    ultralytics_run_name = "_".join(run_name_parts)
    mlflow_run_name = ultralytics_run_name

    active_mlflow_run = mlflow.active_run()
    if active_mlflow_run is not None:
        print(
            f"Warning: An active MLflow run ({active_mlflow_run.info.run_id}) was found before starting a new one. Ending it."
        )
        mlflow.end_run()

    print(f"\n--- Starting Run {i+1}/{len(hyperparameter_combinations)}: {mlflow_run_name} ---")
    print(f"Parameters: {params}")

    with mlflow.start_run(run_name=mlflow_run_name) as run:
        # This is the run object for the current run
        # print(f"MLflow Run ID for this iteration: {run.info.run_id}") # For debugging

        mlflow.log_param("pretrained_model", PRETRAINED_MODEL_NAME)
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("data_yaml", DATA_YAML)
        mlflow.log_param("device_used", device_to_use)

        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        try:
            model = YOLO(PRETRAINED_MODEL_NAME)
            model.train(
                data=DATA_YAML,
                imgsz=IMAGE_SIZE,
                project=ULTRALYTICS_OUTPUT_DIR,
                name=ultralytics_run_name,
                device=device_to_use,
                exist_ok=True,
                **params
            )

            run_output_dir = os.path.join(
                ULTRALYTICS_OUTPUT_DIR, ultralytics_run_name
            )
            best_model_path = os.path.join(run_output_dir, "weights", "best.pt")

            if os.path.exists(best_model_path):
                mlflow.log_artifact(
                    best_model_path, artifact_path="yolo_model_weights"
                )
            else:
                print(
                    f"Warning: Best model not found at {best_model_path} for run {mlflow_run_name}"
                )

            artifacts_to_log = [
                "results.csv", "confusion_matrix.png", "F1_curve.png",
                "PR_curve.png", "P_curve.png", "R_curve.png", "labels.jpg",
                "labels_correlogram.jpg", "val_batch0_labels.jpg",
                "val_batch0_pred.jpg",
            ]
            for item_name in artifacts_to_log:
                item_path = os.path.join(run_output_dir, item_name)
                if os.path.exists(item_path):
                    mlflow.log_artifact(
                        item_path, artifact_path="training_outputs"
                    )
                else:
                    print(
                        f"Info: Artifact '{item_name}' not found at {item_path} for logging in run {mlflow_run_name}."
                    )
            mlflow.set_tag("run_status", "completed") # Use set_tag for status
            print(f"--- Finished Run: {mlflow_run_name} ---")

        except Exception as e:
            print(f"!!! Error during run {mlflow_run_name}: {e} !!!")
            # Ensure tags/params are logged even on error, if possible (run is active)
            try:
                mlflow.set_tag("run_status", "failed")
                mlflow.log_param("error_message", str(e)) # Log error as param
            except Exception as log_err:
                print(f"Error logging failure status to MLflow: {log_err}")
            continue

print("\nGrid search finished. Check MLflow UI for results.")