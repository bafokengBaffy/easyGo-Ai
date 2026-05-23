from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/price-predict', tags=['price_predict'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='price_predict',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
