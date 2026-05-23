from typing import AsyncGenerator
from fastapi import Depends
from .middleware import get_request_id
from ..config import settings

def get_settings():
    return settings

def request_context() -> AsyncGenerator[str, None]:
    request_id = get_request_id()
    yield request_id

def common_dependencies(settings=Depends(get_settings)):
    return {"settings": settings}
