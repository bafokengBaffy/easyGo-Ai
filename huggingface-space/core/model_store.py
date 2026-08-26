from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib

from core.config import AppConfig, model_file_map
from core.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    name: str
    path: Path
    size_bytes: int
    checksum_sha256: str
    model_type: str


class ModelStore:
    def __init__(self, config: AppConfig):
        self._config = config
        self._models: Dict[str, Any] = {}
        self._infos: Dict[str, ModelInfo] = {}

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _safe_model_type(model: Any) -> str:
        return f"{model.__class__.__module__}.{model.__class__.__name__}"

    def load_all(self) -> None:
        mapping = model_file_map()
        missing: list[str] = []

        logger.info("loading_models", extra={"count": len(mapping)})
        for name, filename in mapping.items():
            path = self._config.model_dir / filename
            if not path.exists():
                missing.append(str(path))
                continue

            model = joblib.load(path)
            info = ModelInfo(
                name=name,
                path=path,
                size_bytes=path.stat().st_size,
                checksum_sha256=self._sha256(path),
                model_type=self._safe_model_type(model),
            )
            self._models[name] = model
            self._infos[name] = info
            logger.info(
                "model_loaded",
                extra={
                    "model": name,
                    "size_bytes": info.size_bytes,
                    "type": info.model_type,
                },
            )

        if missing:
            raise RuntimeError(f"Missing model files: {missing}")

    def model(self, name: str) -> Any | None:
        return self._models.get(name)

    def metadata(self, name: str) -> ModelInfo | None:
        return self._infos.get(name)

    def all_metadata(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name, info in self._infos.items():
            out[name] = {
                "path": str(info.path),
                "size_bytes": info.size_bytes,
                "checksum_sha256": info.checksum_sha256,
                "model_type": info.model_type,
            }
        return out

    def loaded_names(self) -> list[str]:
        return sorted(self._models.keys())

