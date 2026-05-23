from fastapi import APIRouter
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix='/demand-forecast', tags=['demand_forecast'])

@router.post('/', response_model=PredictionResponse)
async def handle(request: PredictionRequest):
    return PredictionResponse(
        model='demand_forecast',
        prediction={'result': 'placeholder'},
        confidence=0.85,
    )
