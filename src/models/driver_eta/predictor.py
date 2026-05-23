from pathlib import Path
import pandas as pd
from ..driver_eta.model import DriverETAModel
from ...models import MODEL_DIR

MODEL_PATH = MODEL_DIR / "driver_eta.joblib"

def predict(df: pd.DataFrame):
    features = ["trips_per_day", "fast_responder", "eta_abs_error"]
    model = DriverETAModel()
    model.load(MODEL_PATH)
    return model.predict(df[features])
