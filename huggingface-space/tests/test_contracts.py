from core.contracts import (
    get_contract,
    normalize_and_validate_values,
    ordered_features,
    validate_feature_presence,
)


def test_missing_required_features_detected():
    contract = get_contract("driver_eta")
    issues = validate_feature_presence(contract, {"distance_km": 3.0})
    assert issues


def test_bounds_validation():
    contract = get_contract("driver_eta")
    normalized, issues = normalize_and_validate_values(
        contract,
        {
            "distance_km": -1,
            "traffic_index": 3,
            "hour_of_day": 12,
            "is_weekend": 0,
            "weather_severity": 1,
            "driver_experience_years": 10,
            "pickup_zone_density": 0.4,
        },
    )
    assert "distance_km" not in normalized
    assert issues


def test_ordered_features_puts_contract_first():
    contract = get_contract("rider_churn")
    normalized = {
        "trip_count_30d": 1,
        "age": 2,
        "avg_fare": 3,
        "days_since_last_ride": 4,
        "cancellation_rate": 5,
        "support_tickets_90d": 6,
        "rating": 4.9,
        "custom": 10,
    }
    ordered = ordered_features(contract, normalized)
    keys = list(ordered.keys())
    assert keys[0] == "age"
    assert "custom" in ordered

