from fastapi import FastAPI
from .middleware import configure_middleware
from .v1 import api_router

def create_app() -> FastAPI:
    app = FastAPI(title="easyGo AI Services", version="1.0.0")
    configure_middleware(app)
    app.include_router(api_router, prefix="/api/v1")
    return app

app = create_app()
