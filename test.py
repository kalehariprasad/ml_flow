import dagshub

import mlflow
dagshub.init(repo_owner='kalehariprasad', repo_name='ml_flow', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/kalehariprasad/ml_flow.mlflow')
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)