# easyGo AI Services

This package implements the AI and ML service layer for easyGo. It includes:

- FastAPI REST endpoints for inference and metadata
- feature engineering and data loaders
- model training and evaluation pipelines
- explainability, monitoring, and deployment support

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

The web backend calls this service server-to-server through `AI_SERVICE_URL`.
Set `AI_SERVICE_API_KEY` in both services to enable the optional service-key check.
For direct browser tooling only, set `AI_CORS_ORIGINS` to a comma-separated list
of trusted origins. Production web and mobile clients should use the web backend
gateway at `/api/v1/ai` instead of exposing the AI service directly.
