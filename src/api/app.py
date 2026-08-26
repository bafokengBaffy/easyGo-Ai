from fastapi import FastAPI
import logging
import sys
from .middleware import configure_middleware
from .v1 import api_router

def create_app() -> FastAPI:
    app = FastAPI(title="easyGo AI Services", version="1.0.0")
    configure_middleware(app)
    app.include_router(api_router, prefix="/api/v1")

    @app.on_event("startup")
    async def log_endpoint_catalog() -> None:
        logger = logging.getLogger("easygo.ai.startup")
        logger.info("AI endpoint catalog:")
        for route in sorted(app.routes, key=lambda item: item.path):
            methods = ",".join(sorted(route.methods or []))
            if methods:
                logger.info("  %s %s", methods, route.path)

    return app

app = create_app()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
