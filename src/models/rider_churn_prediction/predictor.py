from pathlib import Path
import pandas as pd
from ..rider_churn_prediction.model import RiderChurnModel
from ...models import MODEL_DIR

MODEL_PATH = MODEL_DIR / "rider_churn.joblib"

def predict(df: pd.DataFrame):
    features = ["trips_per_week", "is_new", "distance_per_trip", "rating_scaled"]
    model = RiderChurnModel()
    model.load(MODEL_PATH)
    return model.predict(df[features])
