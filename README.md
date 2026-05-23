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
