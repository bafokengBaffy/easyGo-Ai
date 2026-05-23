from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/driver-rating', tags=['driver_rating'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='driver_rating',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
