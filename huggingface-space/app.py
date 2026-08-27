from __future__ import annotations

import json
import logging
import os
import time

import gradio as gr

from core.config import load_config
from core.errors import AppError
from core.feature_adapter import adapt_features_for_model
from core.inference_service import InferenceService
from core.logging_utils import configure_logging, get_logger
from core.model_store import ModelStore
from core.contracts import get_contract


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = load_config(__import__("pathlib").Path(BASE_DIR))
configure_logging(CONFIG.log_level)
LOGGER = get_logger(__name__)
MODEL_STORE = ModelStore(CONFIG)
MODEL_STORE.load_all()
INFERENCE_SERVICE = InferenceService(CONFIG, MODEL_STORE)
MODEL_NAMES = ["rider_churn", "rider_ltv", "driver_eta", "driver_acceptance"]


def model_status() -> str:
    loaded = MODEL_STORE.loaded_names()
    missing = [name for name in MODEL_NAMES if name not in loaded]
    if not missing:
        return "All four trained models are loaded."
    return f"Loaded models: {', '.join(loaded) or 'none'}. Missing artifacts: {', '.join(missing)}."


def feature_template(model_name: str) -> str:
    return json.dumps({feature.name: feature.default or 0 for feature in get_contract(model_name).features}, indent=2)


def predict(model_name: str, feature_json: str) -> dict:
    started_at = time.perf_counter()
    try:
        features = json.loads(feature_json or "{}")
        if not isinstance(features, dict):
            raise ValueError("Features must be a JSON object.")
        result = INFERENCE_SERVICE.predict(model_name, features)
        response = {
            "model": result.model,
            "prediction": result.prediction,
            "probabilities": result.probabilities,
            "latency_ms": result.latency_ms,
            "features_used": result.features_used,
            "warnings": result.warnings,
        }
        LOGGER.info("gradio_prediction_complete", extra={"model": model_name, "duration_ms": round((time.perf_counter() - started_at) * 1000, 2)})
        return response
    except AppError as error:
        LOGGER.warning("gradio_prediction_failed", extra={"model": model_name, "error": error.message})
        return {"error": error.to_dict()["error"]}
    except Exception as error:
        LOGGER.exception("gradio_prediction_invalid", extra={"model": model_name})
        return {"error": {"code": "invalid_request", "message": str(error)}}


with gr.Blocks(title="easygoAI") as demo:
    gr.Markdown("# easygoAI\nEasyGo model inference")
    status = gr.Markdown(model_status())
    model = gr.Dropdown(MODEL_NAMES, value=MODEL_NAMES[0], label="Model")
    features = gr.Code(value=feature_template(MODEL_NAMES[0]), language="json", label="Features JSON")
    model.change(feature_template, inputs=model, outputs=features)
    run = gr.Button("Run prediction", variant="primary")
    result = gr.JSON(label="Prediction")
    run.click(predict, inputs=[model, features], outputs=result)
    refresh = gr.Button("Refresh model status")
    refresh.click(lambda: model_status(), outputs=status)


LOGGER.info("gradio_endpoint_catalog")
LOGGER.info("  GET /")
LOGGER.info("  POST /gradio_api/call/predict")
LOGGER.info("  model status: %s", model_status())

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
