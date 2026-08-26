from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.deps import RateLimiter
from api.routes import build_router
from core.config import load_config
from core.errors import AppError
from core.inference_service import InferenceService
from core.logging_utils import configure_logging, get_logger
from core.model_store import ModelStore


BASE_DIR = Path(__file__).resolve().parent
CONFIG = load_config(BASE_DIR)
configure_logging(CONFIG.log_level)
LOGGER = get_logger(__name__)

MODEL_STORE = ModelStore(CONFIG)
MODEL_STORE.load_all()
INFERENCE_SERVICE = InferenceService(CONFIG, MODEL_STORE)
RATE_LIMITER = RateLimiter(limit_per_minute=CONFIG.rate_limit_per_minute)


def create_app() -> FastAPI:
    app = FastAPI(
        title=CONFIG.app_name,
        version=CONFIG.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CONFIG.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("unhandled_exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An internal server error occurred",
                    "details": {},
                }
            },
        )

    app.include_router(
        build_router(
            config=CONFIG,
            service=INFERENCE_SERVICE,
            model_store=MODEL_STORE,
            limiter=RATE_LIMITER,
        )
    )
    return app


app = create_app()

