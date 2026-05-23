from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from ..driver_acceptance.model import DriverAcceptanceModel
from ...data.loaders.csv_loader import load_csv
from ...data.processors.cleaning import basic_cleaning
from ...data.features.feature_engineering import driver_features
from ...models import MODEL_DIR

def train(save_name: str = "driver_acceptance.joblib") -> None:
    root = Path(__file__).resolve().parents[3]
    data_path = root / "data" / "raw" / "driver_dataset.csv"
    df = load_csv(data_path)
    df = basic_cleaning(df)
    df = driver_features(df)
    # create a synthetic binary target from acceptance_rate
    df["accepted"] = (df["acceptance_rate"] >= 0.8).astype(int)
    features = ["trips_per_day", "fast_responder", "avg_response_time_s"]
    df = df.dropna(subset=features + ["accepted"])
    X = df[features]
    y = df["accepted"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DriverAcceptanceModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    try:
        auc = roc_auc_score(y_test, preds)
    except Exception:
        auc = 0.0
    print(f"Driver acceptance model AUC: {auc:.4f}")
    out = MODEL_DIR / save_name
    model.save(out)
    print(f"Saved driver acceptance model to {out}")

if __name__ == '__main__':
    train()
