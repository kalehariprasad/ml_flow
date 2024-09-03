import mlflow
import pandas as pd
import mlflow.sklearn
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (r2_score,mean_absolute_error,mean_squared_error)
import dagshub
dagshub.init(repo_owner='kalehariprasad', repo_name='ml_flow', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/kalehariprasad/ml_flow.mlflow')



# Load data
df = pd.read_csv("diabetes.csv")
x = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize model and GridSearchCV
rf = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30]
}

gridsearch = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=1, verbose=2)

# Start MLflow experiment
mlflow.set_experiment('mlflow gridsearch with random forest')
with mlflow.start_run():
    gridsearch.fit(x_train, y_train)

    # Log all parameter sets and their corresponding metrics
    for i in range(len(gridsearch.cv_results_['params'])):
        params = gridsearch.cv_results_['params'][i]
        mean_test_score = gridsearch.cv_results_['mean_test_score'][i]
        
        # Log parameters
        mlflow.log_params(params)
        
        # Log metric
        mlflow.log_metric('mean_test_score', mean_test_score, step=i)

    best_params = gridsearch.best_params_
    best_score = gridsearch.best_score_

    # Log best parameters and metrics
    mlflow.log_params(best_params)
    mlflow.log_metric("Best Score", best_score)
    
    # Save and log training and testing dataframes as CSV files
    train_df = x_train.copy()
    train_df['Outcome'] = y_train.values
    train_df.to_csv("train_df.csv", index=False)
    mlflow.log_artifact("train_df.csv")
    
    test_df = x_test.copy()
    test_df['Outcome'] = y_test.values
    test_df.to_csv("test_df.csv", index=False)
    mlflow.log_artifact("test_df.csv")
    
    # Log model
    mlflow.sklearn.log_model(gridsearch.best_estimator_, "model")
    mlflow.set_tag('author', "hari")  # Changed 'authon' to 'author' for clarity

    print(best_params)
    print(best_score)
