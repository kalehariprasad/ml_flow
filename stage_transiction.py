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

client=MlflowClient()


model_name="diabetes-RF"

# Get the latest version of the model
def get_latest_model_version(model_name):
    model_versions = client.get_model_versions(model_name)
    if model_versions:
        # Sort the versions and get the latest one
        latest_version = max(int(version.version) for version in model_versions)
        return latest_version
    else:
        return None

latest_version = get_latest_model_version(model_name)

if latest_version is not None:
    new_stage = "Staging"
    model_name="diabetes-RF"
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage=new_stage,
        archive_existing_versions=False
    )

    print(f"{latest_version}rd version of {model_name} transitioned to {new_stage}")
