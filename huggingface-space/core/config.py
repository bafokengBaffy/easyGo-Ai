from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class AppConfig:
    app_name: str
    app_version: str
    environment: str
    model_dir: Path
    request_timeout_seconds: int
    max_feature_count: int
    enforce_contracts: bool
    api_key_enabled: bool
    api_key: str
    rate_limit_enabled: bool
    rate_limit_per_minute: int
    cors_allow_origins: list[str]
    log_level: str

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def load_config(base_dir: Path) -> AppConfig:
    model_dir = Path(os.getenv("MODEL_DIR", str(base_dir / "models_artifacts")))
    model_dir = model_dir.resolve()

    return AppConfig(
        app_name=os.getenv("APP_NAME", "easygoAI"),
        app_version=os.getenv("APP_VERSION", "2.0.0"),
        environment=os.getenv("ENVIRONMENT", "production"),
        model_dir=model_dir,
        request_timeout_seconds=_get_int("REQUEST_TIMEOUT_SECONDS", 10),
        max_feature_count=_get_int("MAX_FEATURE_COUNT", 256),
        enforce_contracts=_get_bool("ENFORCE_CONTRACTS", True),
        api_key_enabled=_get_bool("API_KEY_ENABLED", False),
        api_key=os.getenv("API_KEY", ""),
        rate_limit_enabled=_get_bool("RATE_LIMIT_ENABLED", True),
        rate_limit_per_minute=_get_int("RATE_LIMIT_PER_MINUTE", 120),
        cors_allow_origins=_get_list("CORS_ALLOW_ORIGINS", "*"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


def model_file_map() -> Dict[str, str]:
    return {
        "rider_churn": "rider_churn.joblib",
        "rider_ltv": "rider_ltv.joblib",
        "driver_eta": "driver_eta.joblib",
        "driver_acceptance": "driver_acceptance.joblib",
    }

