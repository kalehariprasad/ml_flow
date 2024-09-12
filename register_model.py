from mlflow.tracking import MlflowClient
import mlflow
client=MlflowClient()
import time

run_id='1de2a7c27a6d4378b9bb5a227dc02c59'
model_path="mlflow-artifacts:/6e3fee7ee2e0452b8bfe35f65866b7a4/1de2a7c27a6d4378b9bb5a227dc02c59/artifacts/random_forest"
model_uri=f"runs/{run_id}/{model_path}"
model_name="diabetes-RF-1"

result=mlflow.register_model(model_uri,model_name)

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