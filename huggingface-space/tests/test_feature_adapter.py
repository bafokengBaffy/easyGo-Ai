from core.feature_adapter import adapt_features_for_model


class DummyModel:
    def __init__(self, features):
        self.feature_names_in_ = features


def test_adapter_keeps_present_features():
    model = DummyModel(["a", "b"])
    out = adapt_features_for_model(model, {"a": 1.0, "b": 2.0})
    assert out == {"a": 1.0, "b": 2.0}


def test_adapter_maps_aliases():
    model = DummyModel(["distance_km", "fare_amount"])
    out = adapt_features_for_model(
        model,
        {"distance_to_pickup_km": 11.0, "avg_fare": 99.0},
    )
    assert out["distance_km"] == 11.0
    assert out["fare_amount"] == 99.0


def test_adapter_fills_missing_with_zero():
    model = DummyModel(["x", "y", "z"])
    out = adapt_features_for_model(model, {"x": 3})
    assert out == {"x": 3.0, "y": 0.0, "z": 0.0}

