from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.config import AppConfig
from core.contracts import (
    get_contract,
    normalize_and_validate_values,
    ordered_features,
    validate_feature_presence,
)
from core.errors import ContractValidationError, ModelNotLoadedError, PredictionError
from core.feature_adapter import adapt_features_for_model
from core.logging_utils import get_logger
from core.model_store import ModelStore


logger = get_logger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    prediction: float
    probabilities: list[float] | None
    model: str
    latency_ms: float
    features_used: list[str]
    warnings: list[str]


class InferenceService:
    def __init__(self, config: AppConfig, model_store: ModelStore):
        self._config = config
        self._store = model_store

    def _to_float(self, value: Any) -> float:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            raise ValueError("nan_or_inf")
        return out

    def _validate_contract(self, model_name: str, features: dict[str, float]) -> dict[str, float]:
        contract = get_contract(model_name)
        issues = validate_feature_presence(contract, features)

        normalized, value_issues = normalize_and_validate_values(contract, features)
        issues.extend(value_issues)

        if self._config.enforce_contracts and issues:
            raise ContractValidationError(model_name=model_name, reason="; ".join(issues))
        return ordered_features(contract, normalized)

    def predict(self, model_name: str, features: dict[str, float]) -> PredictionResult:
        if len(features) > self._config.max_feature_count:
            raise ContractValidationError(
                model_name=model_name,
                reason=f"feature_count={len(features)} exceeds max={self._config.max_feature_count}",
            )

        model = self._store.model(model_name)
        if model is None:
            raise ModelNotLoadedError(model_name)

        validated = self._validate_contract(model_name, features)
        row = {k: self._to_float(v) for k, v in validated.items()}
        row = adapt_features_for_model(model, row)
        frame = pd.DataFrame([row])

        start = time.perf_counter()
        try:
            pred = model.predict(frame)
            prediction_value = float(pred[0])
        except Exception as exc:  # pragma: no cover
            raise PredictionError(model_name, str(exc)) from exc

        probabilities: list[float] | None = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(frame)[0]
                probabilities = [float(x) for x in proba]
            except Exception:
                probabilities = None

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "prediction_complete",
            extra={
                "model": model_name,
                "latency_ms": round(latency_ms, 3),
                "feature_count": len(row),
            },
        )

        warnings: list[str] = []
        if probabilities and len(probabilities) == 2:
            confidence = max(probabilities)
            if confidence < 0.55:
                warnings.append("low_confidence_binary_prediction")

        return PredictionResult(
            prediction=prediction_value,
            probabilities=probabilities,
            model=model_name,
            latency_ms=latency_ms,
            features_used=list(row.keys()),
            warnings=warnings,
        )
