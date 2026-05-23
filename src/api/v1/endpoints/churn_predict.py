from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/churn-predict', tags=['churn_predict'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='churn_predict',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
