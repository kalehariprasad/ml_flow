import numpy as np
import mlflow.pyfunc
import dagshub
import mlflow
import pandas as pd
import os 
import time
from requests.exceptions import RequestException
from mlflow.tracking import MlflowClient
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
    registered_models = client.search_registered_models(filter_string=f"name='{model_name}'")
    if registered_models:
        # Assuming model_name is unique and we take the first result
        model = registered_models[0]
        # Get the latest version from the latest_versions property
        if model.latest_versions:
            latest_version = max(int(version.version) for version in model.latest_versions)
            return latest_version
    return None


latest_version = get_latest_model_version(model_name)
model=mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{latest_version}")


def load_model_with_retry(model_uri, retries=3, delay=5):
    for attempt in range(retries):
        try:
            model = mlflow.pyfunc.load_model(model_uri=model_uri)
            return model
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    raise RuntimeError("Failed to load model after several attempts")

model = load_model_with_retry(f"models:/{model_name}/{latest_version}")

# Define feature names and expected types
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
feature_types = {'Pregnancies': np.int64, 'Glucose': np.int64, 'BloodPressure': np.int64, 'SkinThickness': np.int64, 'Insulin': np.int64, 'BMI': np.float64, 'DiabetesPedigreeFunction': np.float64, 'Age': np.int64}

# Define input data
input_data = {
    'Pregnancies': [1],
    'Glucose': [85],
    'BloodPressure': [66],
    'SkinThickness': [29],
    'Insulin': [0],
    'BMI': [26.6],
    'DiabetesPedigreeFunction': [0.351],
    'Age': [31]
}

# Convert to DataFrame with correct types
df = pd.DataFrame(input_data)
for feature, dtype in feature_types.items():
    df[feature] = df[feature].astype(dtype)
prediction = model.predict(df)

print("Prediction:", prediction)