from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/recommendations', tags=['recommendations'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='recommendations',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
