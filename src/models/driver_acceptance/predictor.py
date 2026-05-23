from pathlib import Path
import pandas as pd
from ..driver_acceptance.model import DriverAcceptanceModel
from ...models import MODEL_DIR

MODEL_PATH = MODEL_DIR / "driver_acceptance.joblib"

def predict(df: pd.DataFrame):
    features = ["trips_per_day", "fast_responder", "avg_response_time_s"]
    model = DriverAcceptanceModel()
    model.load(MODEL_PATH)
    return model.predict(df[features])
