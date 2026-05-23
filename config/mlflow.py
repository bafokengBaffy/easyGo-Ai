import mlflow
from .settings import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow_client = mlflow
