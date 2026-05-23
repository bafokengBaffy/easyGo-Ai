from typing import Sequence

def calculate_metrics(true_values: Sequence[float], predictions: Sequence[float]) -> dict[str, float]:
    if not true_values:
        return {"mse": 0.0}
    mse = sum((t - p) ** 2 for t, p in zip(true_values, predictions)) / len(true_values)
    return {"mse": mse}
