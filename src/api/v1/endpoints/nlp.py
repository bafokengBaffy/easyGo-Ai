from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/nlp', tags=['nlp'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='nlp',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
