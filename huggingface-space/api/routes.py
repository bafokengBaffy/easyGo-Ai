from __future__ import annotations

import uuid
from typing import Callable

from fastapi import APIRouter, Depends, Header, Request

from api.deps import RateLimiter, enforce_rate_limit, require_api_key
from api.schemas import (
    HealthResponse,
    ModelCatalogItem,
    PredictRequest,
    PredictionResponse,
)
from core.config import AppConfig
from core.inference_service import InferenceService
from core.logging_utils import endpoint_ctx, request_id_ctx
from core.model_store import ModelStore


def _identity_from_request(request: Request, x_forwarded_for: str | None) -> str:
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _predict_endpoint(
    model_name: str,
    service: InferenceService,
    config: AppConfig,
    limiter: RateLimiter,
) -> Callable:
    def _handler(
        payload: PredictRequest,
        request: Request,
        x_api_key: str | None = Header(default=None),
        x_forwarded_for: str | None = Header(default=None),
        x_request_id: str | None = Header(default=None),
    ) -> PredictionResponse:
        request_id = x_request_id or str(uuid.uuid4())
        request_id_ctx.set(request_id)
        endpoint_ctx.set(f"/predict/{model_name}")

        require_api_key(config, x_api_key)
        identity = _identity_from_request(request, x_forwarded_for)
        enforce_rate_limit(config, limiter, identity)

        result = service.predict(model_name=model_name, features=payload.features)
        return PredictionResponse(
            model=result.model,
            prediction=result.prediction,
            probabilities=result.probabilities,
            latency_ms=result.latency_ms,
            features_used=result.features_used,
            warnings=result.warnings,
        )

    return _handler


def build_router(
    config: AppConfig,
    service: InferenceService,
    model_store: ModelStore,
    limiter: RateLimiter,
) -> APIRouter:
    router = APIRouter()

    @router.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {"message": f"{config.app_name} running"}

    @router.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            environment=config.environment,
            loaded_models=model_store.loaded_names(),
        )

    @router.get("/ready", tags=["meta"])
    def readiness() -> dict[str, object]:
        loaded = model_store.loaded_names()
        return {"ready": len(loaded) > 0, "loaded_models": loaded}

    @router.get("/models", response_model=list[ModelCatalogItem], tags=["meta"])
    def models() -> list[ModelCatalogItem]:
        output: list[ModelCatalogItem] = []
        for name, info in model_store.all_metadata().items():
            output.append(
                ModelCatalogItem(
                    model=name,
                    path=info["path"],
                    size_bytes=info["size_bytes"],
                    checksum_sha256=info["checksum_sha256"],
                    model_type=info["model_type"],
                )
            )
        return output

    predict_rider_churn = _predict_endpoint("rider_churn", service, config, limiter)
    predict_rider_ltv = _predict_endpoint("rider_ltv", service, config, limiter)
    predict_driver_eta = _predict_endpoint("driver_eta", service, config, limiter)
    predict_driver_acceptance = _predict_endpoint("driver_acceptance", service, config, limiter)

    router.add_api_route(
        path="/predict/rider_churn",
        endpoint=predict_rider_churn,
        methods=["POST"],
        response_model=PredictionResponse,
        tags=["predict"],
    )
    router.add_api_route(
        path="/predict/rider_ltv",
        endpoint=predict_rider_ltv,
        methods=["POST"],
        response_model=PredictionResponse,
        tags=["predict"],
    )
    router.add_api_route(
        path="/predict/driver_eta",
        endpoint=predict_driver_eta,
        methods=["POST"],
        response_model=PredictionResponse,
        tags=["predict"],
    )
    router.add_api_route(
        path="/predict/driver_acceptance",
        endpoint=predict_driver_acceptance,
        methods=["POST"],
        response_model=PredictionResponse,
        tags=["predict"],
    )

    return router
