from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/fraud-detect', tags=['fraud_detect'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='fraud_detect',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
