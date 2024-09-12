import numpy as np
import mlflow.pyfunc
import dagshub
import mlflow
import pandas as pd

dagshub.init(repo_owner='kalehariprasad', repo_name='ml_flow', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/kalehariprasad/ml_flow.mlflow')

model_name="diabetes-RF"
version=1
model=mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{version}")

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