from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/driver-match', tags=['driver_match'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='driver_match',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
