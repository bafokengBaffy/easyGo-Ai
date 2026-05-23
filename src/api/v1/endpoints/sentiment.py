from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/sentiment', tags=['sentiment'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='sentiment',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
