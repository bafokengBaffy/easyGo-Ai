from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ..rider_ltv.model import RiderLTVModel
from ...data.loaders.csv_loader import load_csv
from ...data.processors.cleaning import basic_cleaning
from ...data.features.feature_engineering import rider_features
from ...models import MODEL_DIR

def train(save_name: str = "rider_ltv.joblib") -> None:
    root = Path(__file__).resolve().parents[3]
    data_path = root / "data" / "raw" / "rider_dataset.csv"
    df = load_csv(data_path)
    df = basic_cleaning(df)
    df = rider_features(df)
    features = ["trips_per_week", "is_new", "distance_per_trip", "rating_scaled"]
    df = df.dropna(subset=features + ["estimated_ltv"])
    X = df[features]
    y = df["estimated_ltv"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RiderLTVModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Rider LTV model MSE: {mse:.4f}")
    out = MODEL_DIR / save_name
    model.save(out)
    print(f"Saved rider ltv model to {out}")

if __name__ == '__main__':
    train()
