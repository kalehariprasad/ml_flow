from mlflow.tracking import MlflowClient
import mlflow
import time
import dagshub

dagshub.init(repo_owner='kalehariprasad', repo_name='ml_flow', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/kalehariprasad/ml_flow.mlflow')

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