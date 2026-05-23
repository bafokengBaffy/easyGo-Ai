from fastapi import APIRouter
from .v1.endpoints import (
    batch_predict,
    churn_predict,
    compliance,
    demand_forecast,
    driver_match,
    driver_rating,
    eta_predict,
    feedback,
    fraud_detect,
    health,
    image_analysis,
    models_metadata,
    nlp,
    price_predict,
    recommendations,
    risk_score,
    route_optimize,
    sentiment,
    surge_pricing,
    version,
)

api_router = APIRouter()

api_router.include_router(batch_predict.router)
api_router.include_router(churn_predict.router)
api_router.include_router(compliance.router)
api_router.include_router(demand_forecast.router)
api_router.include_router(driver_match.router)
api_router.include_router(driver_rating.router)
api_router.include_router(eta_predict.router)
api_router.include_router(feedback.router)
api_router.include_router(fraud_detect.router)
api_router.include_router(health.router)
api_router.include_router(image_analysis.router)
api_router.include_router(models_metadata.router)
api_router.include_router(nlp.router)
api_router.include_router(price_predict.router)
api_router.include_router(recommendations.router)
api_router.include_router(risk_score.router)
api_router.include_router(route_optimize.router)
api_router.include_router(sentiment.router)
api_router.include_router(surge_pricing.router)
api_router.include_router(version.router)
