from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/route-optimize', tags=['route_optimize'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='route_optimize',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
