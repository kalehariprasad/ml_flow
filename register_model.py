from mlflow.tracking import MlflowClient
import mlflow
import time
import dagshub
import os

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "kalehariprasad"
repo_name = "ml_flow"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
client = MlflowClient()

# Get the latest run ID
def get_latest_run_id(experiment_name):
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.list_runs(experiment.experiment_id, order_by=["start_time desc"])
    if runs:
        return runs[0].info.run_id
    else:
        return None

# Specify your experiment name
experiment_name = "diabetes-rf-hp"
latest_run_id = get_latest_run_id(experiment_name)

if latest_run_id:

    model_path="mlflow-artifacts:/6e3fee7ee2e0452b8bfe35f65866b7a4/1de2a7c27a6d4378b9bb5a227dc02c59/artifacts/random_forest"
    model_name="diabetes-RF"
    model_uri = f"runs:/{latest_run_id}/{model_path}"
    result=mlflow.register_model(model_uri=model_uri, name=model_name)

time.sleep(5)
client.update_model_version(
    name=model_name,
    version=result.version,
    description="model is created for registering model through code"
)


client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key='created by',
    value="Hari Prasad")