from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/eta-predict', tags=['eta_predict'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='eta_predict',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
