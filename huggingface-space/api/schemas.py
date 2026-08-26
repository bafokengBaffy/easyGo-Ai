from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    model: str
    prediction: float
    probabilities: list[float] | None = None
    latency_ms: float
    features_used: list[str]
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    environment: str
    loaded_models: list[str]


class ModelCatalogItem(BaseModel):
    model: str
    path: str
    size_bytes: int
    checksum_sha256: str
    model_type: str


class ErrorEnvelope(BaseModel):
    error: Dict[str, Any]

