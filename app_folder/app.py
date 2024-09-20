import numpy as np
import mlflow.pyfunc
import dagshub
import mlflow
import pandas as pd
import streamlit as st
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

# Define feature names and expected types
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
feature_types = {'Pregnancies': np.int64, 'Glucose': np.int64, 'BloodPressure': np.int64, 'SkinThickness': np.int64, 'Insulin': np.int64, 'BMI': np.float64, 'DiabetesPedigreeFunction': np.float64, 'Age': np.int64}

# Streamlit inputs
Pregnancies_ = st.number_input("Number of Pregnancies", min_value=0, value=0)
Glucose_ = st.number_input("Glucose level", min_value=0, value=0)
BloodPressure_ = st.number_input("Blood Pressure", min_value=0, value=0)
SkinThickness_ = st.number_input("Skin Thickness", min_value=0, value=0)
Insulin_ = st.number_input("Insulin", min_value=0, value=0)
BMI_ = st.number_input("BMI", min_value=0.0, value=0.0)
DiabetesPedigreeFunction_ = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.0)
Age_ = st.number_input("Age", min_value=0, value=0)

# Predict button
if st.button('Predict'):
    # Prepare the input data
    input_data = {
        'Pregnancies': [Pregnancies_],
        'Glucose': [Glucose_],
        'BloodPressure': [BloodPressure_],
        'SkinThickness': [SkinThickness_],
        'Insulin': [Insulin_],
        'BMI': [BMI_],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction_],
        'Age': [Age_]
    }
    
    # Create DataFrame
    df = pd.DataFrame(input_data)
    
    # Convert data types
    for feature, dtype in feature_types.items():
        df[feature] = df[feature].astype(dtype)
    
    # Make prediction
    prediction = model.predict(df)
    
    # Show result
    if prediction[0] == 1:
        st.write("Prediction: The person is likely to have diabetes.")
    else:
        st.write("Prediction: The person is unlikely to have diabetes.")