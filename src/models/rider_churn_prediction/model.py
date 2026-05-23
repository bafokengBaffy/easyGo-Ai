from typing import Any
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

class RiderChurnModel:
    def __init__(self, model: Any | None = None):
        self.model = model or RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: Path):
        joblib.dump(self.model, path)

    def load(self, path: Path):
        self.model = joblib.load(path)
        return self
