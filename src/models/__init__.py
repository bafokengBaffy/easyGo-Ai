"""Model package registry for easyGo AI services."""
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models_artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

__all__ = ["MODEL_DIR"]
