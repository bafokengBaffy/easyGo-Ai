from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/image-analysis', tags=['image_analysis'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='image_analysis',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
