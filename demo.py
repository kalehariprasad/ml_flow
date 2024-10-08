import mlflow
import os
import pandas as pd
import mlflow.sklearn
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error)
import dagshub

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

# Load data
df = pd.read_csv("diabetes.csv")
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the RandomForestClassifier model
rf = RandomForestClassifier(random_state=42)

# Defining the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [5,50, 100,150, 200],
    'max_depth': [None, 5, 9, 11]
}

# Applying GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

mlflow.set_experiment('diabetes-rf-hp')

with mlflow.start_run(description="Best hyperparameter trained RF model") as parent:
    grid_search.fit(X_train, y_train)

    # log all the children
    for i in range(len(grid_search.cv_results_['params'])):

        with mlflow.start_run(nested=True) as child:

            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_['mean_test_score'][i])

    # Displaying the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # params

    mlflow.log_params(best_params)

    # metrics
    mlflow.log_metric("accuracy", best_score)

    # data
    train_df = X_train.copy()
    train_df['Outcome'] = y_train

    train_df = mlflow.data.from_pandas(train_df)

    mlflow.log_input(train_df, "training")

    test_df = X_test.copy()
    test_df['Outcome'] = y_test

    test_df = mlflow.data.from_pandas(test_df)

    mlflow.log_input(test_df, "training")

    # source code
    mlflow.log_artifact(__file__)

    
    # Infer model signature
    
    signature = mlflow.models.infer_signature(X_train, grid_search.best_estimator_.predict(X_train))
    
    # model
    
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest", signature=signature)

    # tags
    mlflow.set_tag("author","Hari Prasad")

