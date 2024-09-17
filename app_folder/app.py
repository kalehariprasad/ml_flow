import numpy as np
import mlflow.pyfunc
import dagshub
import mlflow
import pandas as pd
import streamlit as st


dagshub.init(repo_owner='kalehariprasad', repo_name='ml_flow', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/kalehariprasad/ml_flow.mlflow')

model_name="diabetes-RF"
version=1
model=mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{version}")

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