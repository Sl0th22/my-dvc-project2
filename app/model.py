import os
import mlflow

tracking_uri = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/Sl0th22/my-dvc-project2.mlflow",
)
experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "RandomForest_V1.0.1")

mlflow.set_tracking_uri(tracking_uri)
client = mlflow.MlflowClient()

experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
  raise ValueError(f"Experience not found : {experiment_name}")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
if not runs:
  raise ValueError(f"No run found in the experience : {experiment_name}")

latest_run = runs[0]
run_id = latest_run.info.run_id
run_name = latest_run.data.tags.get("mlflow.runName", "")

artifacts = client.list_artifacts(run_id)
if not artifacts:
  raise ValueError(f"No artefact found in the run : {run_id}")

model_artifact_path = (
    run_name
    if any(a.path == run_name for a in artifacts)
    else artifacts[0].path
)
model_uri = f"runs:/{run_id}/{model_artifact_path}"

mlflow.artifacts.download_artifacts(
    artifact_uri=model_uri, dst_path="./model_artifact"
)
print(f"Model dowloadded from {model_uri} to ./model_artifact")