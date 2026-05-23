from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/surge-pricing', tags=['surge_pricing'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='surge_pricing',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
