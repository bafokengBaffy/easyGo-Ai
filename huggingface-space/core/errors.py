from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AppError(Exception):
    code: str
    message: str
    status_code: int = 400
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details or {},
            }
        }


class UnauthorizedError(AppError):
    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(code="unauthorized", message=message, status_code=401)


class RateLimitError(AppError):
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(code="rate_limited", message=message, status_code=429)


class ModelNotLoadedError(AppError):
    def __init__(self, model_name: str):
        super().__init__(
            code="model_not_loaded",
            message=f"Model not loaded: {model_name}",
            status_code=500,
            details={"model": model_name},
        )


class ContractValidationError(AppError):
    def __init__(self, model_name: str, reason: str):
        super().__init__(
            code="invalid_features",
            message=f"Feature contract validation failed for model {model_name}",
            status_code=422,
            details={"model": model_name, "reason": reason},
        )


class PredictionError(AppError):
    def __init__(self, model_name: str, reason: str):
        super().__init__(
            code="prediction_failed",
            message=f"Prediction failed for model {model_name}",
            status_code=500,
            details={"model": model_name, "reason": reason},
        )

