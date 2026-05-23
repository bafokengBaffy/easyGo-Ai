from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/risk-score', tags=['risk_score'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='risk_score',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
