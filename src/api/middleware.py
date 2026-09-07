from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import HTTPException
import os
import logging
import time

request_ids: dict[str, str] = {}
logger = logging.getLogger("easygo.ai.requests")

def configure_middleware(app: FastAPI) -> None:
    configured_origins = [origin.strip() for origin in os.getenv("AI_CORS_ORIGINS", "").split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=configured_origins or ["*"],
        allow_credentials=bool(configured_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        started_at = time.perf_counter()
        request_id = request.headers.get("X-Request-ID", "unknown")
        request.state.request_id = request_id
        configured_key = os.getenv("AI_SERVICE_API_KEY", "").strip()
        provided_key = request.headers.get("X-AI-Service-Key", "")
        if configured_key and provided_key != configured_key:
            return JSONResponse(status_code=401, content={"detail": "Invalid AI service key"})
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.exception(
                "AI %s %s -> 500 (%.2f ms) request_id=%s",
                request.method,
                request.url.path,
                duration_ms,
                request_id,
            )
            raise
        response.headers["X-Request-ID"] = request_id
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "AI %s %s -> %s (%.2f ms) request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
        return response

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

def get_request_id() -> str:
    return request_ids.get("current", "unknown")
