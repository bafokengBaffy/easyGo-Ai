import numpy as np

class EtaPredictionModel:
    def predict(self, features: list[float]) -> float:
        return float(np.mean(features)) if features else 0.0
