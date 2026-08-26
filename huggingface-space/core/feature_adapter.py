from __future__ import annotations

from typing import Any


FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "distance_km": ("distance_to_pickup_km", "trip_distance_km"),
    "fare_amount": ("avg_fare", "fare"),
    "traffic_index": ("traffic_level",),
    "driver_experience_years": ("driver_experience",),
    "pickup_zone_density": ("zone_density",),
    "trip_count_30d": ("rides_30d",),
    "trip_count_90d": ("rides_90d",),
    "days_since_last_ride": ("inactivity_days",),
    "support_tickets_90d": ("support_contacts_90d",),
    "driver_acceptance_rate_30d": ("acceptance_rate_30d",),
}


def _first_present(source: dict[str, float], candidates: tuple[str, ...]) -> float | None:
    for key in candidates:
        if key in source:
            return source[key]
    return None


def adapt_features_for_model(model: Any, features: dict[str, float]) -> dict[str, float]:
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return features

    expected_list = [str(x) for x in expected]
    if not expected_list:
        return features

    adapted: dict[str, float] = {}
    for feature_name in expected_list:
        if feature_name in features:
            adapted[feature_name] = float(features[feature_name])
            continue

        alias_candidates = FEATURE_ALIASES.get(feature_name, ())
        alias_value = _first_present(features, alias_candidates)
        if alias_value is not None:
            adapted[feature_name] = float(alias_value)
            continue

        adapted[feature_name] = 0.0

    return adapted

