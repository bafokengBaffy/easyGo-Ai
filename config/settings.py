from pydantic import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    app_name: str = "easyGo AI Services"
    env: str = "development"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    database_url: str = "postgresql://user:pass@localhost:5432/easygo_ai"
    redis_url: str = "redis://localhost:6379/0"
    kafka_bootstrap_servers: str = "localhost:9092"
    mlflow_tracking_uri: str = "http://localhost:5000"
    sentry_dsn: str | None = None
    prometheus_exporter_port: int = 8001

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
