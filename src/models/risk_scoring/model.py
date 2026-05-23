import numpy as np

class RiskScoringModel:
    def predict(self, features: list[float]) -> float:
        return float(np.mean(features)) if features else 0.0
