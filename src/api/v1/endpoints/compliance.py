from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/compliance', tags=['compliance'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='compliance',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
