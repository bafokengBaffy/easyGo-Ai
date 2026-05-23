from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/feedback', tags=['feedback'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='feedback',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
