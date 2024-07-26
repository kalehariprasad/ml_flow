import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (r2_score,mean_absolute_error,mean_squared_error)
import dagshub

mlflow.set_tracking_uri('https://github.com/kalehariprasad/ml_flow.git')
dagshub.init(repo_owner='kalehariprasad',repo_name=' ml_flow',mlflow=True)



mlflow.set_experiment("demo_exeperiment") 
with mlflow.start_run():  
    db = load_diabetes()
    n_estimators=100
    max_depth=6
    max_features=3
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)    
    rf.fit(X_train, y_train)
    pred=rf.predict(X_test)
    mean_absolute_error=mean_absolute_error(y_test,pred)
    mean_squared_error=mean_squared_error(y_test,pred)
    r2_score=r2_score(y_test,pred)  
    
    metrics = {
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score
    }
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "max_features": max_features
    }

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(rf ,"Random Forest")
