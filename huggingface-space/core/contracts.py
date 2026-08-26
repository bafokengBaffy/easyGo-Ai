from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str = "float"
    required: bool = True
    min_value: float | None = None
    max_value: float | None = None
    default: float | None = None


@dataclass(frozen=True)
class ModelContract:
    model_name: str
    features: tuple[FeatureSpec, ...]
    allow_extra_features: bool = True

    @property
    def required_names(self) -> set[str]:
        return {f.name for f in self.features if f.required}

    @property
    def all_names(self) -> set[str]:
        return {f.name for f in self.features}


CONTRACTS: Dict[str, ModelContract] = {
    "rider_churn": ModelContract(
        model_name="rider_churn",
        features=(
            FeatureSpec("age", min_value=16, max_value=100),
            FeatureSpec("trip_count_30d", min_value=0, max_value=500),
            FeatureSpec("avg_fare", min_value=0, max_value=10000),
            FeatureSpec("days_since_last_ride", min_value=0, max_value=3650),
            FeatureSpec("cancellation_rate", min_value=0, max_value=1),
            FeatureSpec("support_tickets_90d", min_value=0, max_value=1000),
            FeatureSpec("rating", min_value=1, max_value=5),
        ),
        allow_extra_features=True,
    ),
    "rider_ltv": ModelContract(
        model_name="rider_ltv",
        features=(
            FeatureSpec("age", min_value=16, max_value=100),
            FeatureSpec("tenure_months", min_value=0, max_value=240),
            FeatureSpec("trip_count_90d", min_value=0, max_value=2000),
            FeatureSpec("avg_fare", min_value=0, max_value=10000),
            FeatureSpec("promo_usage_rate", min_value=0, max_value=1),
            FeatureSpec("subscription_member", min_value=0, max_value=1),
            FeatureSpec("rating", min_value=1, max_value=5),
        ),
        allow_extra_features=True,
    ),
    "driver_eta": ModelContract(
        model_name="driver_eta",
        features=(
            FeatureSpec("distance_km", min_value=0, max_value=500),
            FeatureSpec("traffic_index", min_value=0, max_value=10),
            FeatureSpec("hour_of_day", min_value=0, max_value=23),
            FeatureSpec("is_weekend", min_value=0, max_value=1),
            FeatureSpec("weather_severity", min_value=0, max_value=10),
            FeatureSpec("driver_experience_years", min_value=0, max_value=60),
            FeatureSpec("pickup_zone_density", min_value=0, max_value=1),
        ),
        allow_extra_features=True,
    ),
    "driver_acceptance": ModelContract(
        model_name="driver_acceptance",
        features=(
            FeatureSpec("distance_to_pickup_km", min_value=0, max_value=500),
            FeatureSpec("estimated_trip_minutes", min_value=1, max_value=2000),
            FeatureSpec("fare_amount", min_value=0, max_value=10000),
            FeatureSpec("surge_multiplier", min_value=1, max_value=10),
            FeatureSpec("driver_rating", min_value=1, max_value=5),
            FeatureSpec("driver_acceptance_rate_30d", min_value=0, max_value=1),
            FeatureSpec("hour_of_day", min_value=0, max_value=23),
        ),
        allow_extra_features=True,
    ),
}


def get_contract(model_name: str) -> ModelContract:
    if model_name not in CONTRACTS:
        raise KeyError(f"Missing contract for model: {model_name}")
    return CONTRACTS[model_name]


def validate_feature_presence(
    contract: ModelContract,
    features: dict[str, float],
) -> list[str]:
    issues: list[str] = []
    missing = sorted(contract.required_names - set(features.keys()))
    if missing:
        issues.append(f"missing_required_features={missing}")

    if not contract.allow_extra_features:
        extras = sorted(set(features.keys()) - contract.all_names)
        if extras:
            issues.append(f"unexpected_features={extras}")
    return issues


def _validate_bounds(feature_name: str, value: float, spec: FeatureSpec) -> str | None:
    if spec.min_value is not None and value < spec.min_value:
        return f"{feature_name}:value={value} lower_than_min={spec.min_value}"
    if spec.max_value is not None and value > spec.max_value:
        return f"{feature_name}:value={value} greater_than_max={spec.max_value}"
    return None


def normalize_and_validate_values(
    contract: ModelContract,
    features: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    issues: list[str] = []
    normalized: dict[str, float] = {}

    spec_by_name = {spec.name: spec for spec in contract.features}
    for key, raw in features.items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            issues.append(f"{key}:not_numeric")
            continue

        spec = spec_by_name.get(key)
        if spec is None:
            normalized[key] = value
            continue

        bound_issue = _validate_bounds(key, value, spec)
        if bound_issue:
            issues.append(bound_issue)
            continue
        normalized[key] = value

    for spec in contract.features:
        if spec.name not in normalized and spec.default is not None:
            normalized[spec.name] = float(spec.default)

    return normalized, issues


def ordered_features(
    contract: ModelContract,
    normalized_features: dict[str, float],
) -> dict[str, float]:
    ordered: dict[str, float] = {}
    for spec in contract.features:
        if spec.name in normalized_features:
            ordered[spec.name] = normalized_features[spec.name]

    for key, value in normalized_features.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def feature_names(contract: ModelContract) -> Iterable[str]:
    for spec in contract.features:
        yield spec.name

