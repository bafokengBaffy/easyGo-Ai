from pathlib import Path
import pandas as pd
from ..rider_ltv.model import RiderLTVModel
from ...models import MODEL_DIR

MODEL_PATH = MODEL_DIR / "rider_ltv.joblib"

def predict(df: pd.DataFrame):
    features = ["trips_per_week", "is_new", "distance_per_trip", "rating_scaled"]
    model = RiderLTVModel()
    model.load(MODEL_PATH)
    return model.predict(df[features])
