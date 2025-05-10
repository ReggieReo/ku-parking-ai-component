import mlflow

# Set tracking URI if your MLflow server is remote, otherwise defaults to local ./mlruns
mlflow.set_tracking_uri("http://127.0.0.1:5000") # Or your remote server

EXPERIMENT_NAME = "YOLOv8_KU_PARKING"
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print(f"MLflow Experiment ID: {experiment.experiment_id}")
print(f"MLflow Experiment Name: {experiment.name}")
