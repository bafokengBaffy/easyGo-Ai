from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/models-metadata', tags=['models_metadata'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='models_metadata',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
