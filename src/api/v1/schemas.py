from pydantic import BaseModel
from typing import Any

class PredictionRequest(BaseModel):
    payload: dict[str, Any]

class PredictionResponse(BaseModel):
    model: str
    prediction: Any
    confidence: float
