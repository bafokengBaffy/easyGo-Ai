from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from ..driver_eta.model import DriverETAModel
from ...data.loaders.csv_loader import load_csv
from ...data.processors.cleaning import basic_cleaning
from ...data.features.feature_engineering import driver_features
from ...models import MODEL_DIR

def train(save_name: str = "driver_eta.joblib") -> None:
    root = Path(__file__).resolve().parents[3]
    data_path = root / "data" / "raw" / "driver_dataset.csv"
    df = load_csv(data_path)
    df = basic_cleaning(df)
    df = driver_features(df)
    features = ["trips_per_day", "fast_responder", "eta_abs_error"]
    df = df.dropna(subset=features + ["eta_error_seconds"])
    X = df[features]
    y = df["eta_error_seconds"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DriverETAModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Driver ETA model MAE: {mae:.4f}")
    out = MODEL_DIR / save_name
    model.save(out)
    print(f"Saved driver eta model to {out}")

if __name__ == '__main__':
    train()
