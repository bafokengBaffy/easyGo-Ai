from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/batch-predict', tags=['batch_predict'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='batch_predict',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
