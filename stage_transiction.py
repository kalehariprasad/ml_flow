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
version=3
new_stage="Staging"

client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage=new_stage,
    archive_existing_versions=False

)

print(f"{version}rd vesrion  of  {model_name} transictined to {new_stage}")