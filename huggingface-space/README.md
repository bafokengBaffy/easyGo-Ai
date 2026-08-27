---
emoji: "🤖"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.49.1"
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: AI/ML models for the EasyGo ride-hailing platform.
---

# easygoAI

Gradio Space for EasyGo ML inference.

## Endpoints
- Model status is shown in the Space interface.
- Predictions are available for rider churn, rider LTV, driver ETA, and driver acceptance.

Startup logs list every registered endpoint. Each request is logged with its
method, path, status code, and duration in the Space logs.

The Space uses scikit-learn 1.7.2, which is compatible with the Python 3.10
runtime used by the Space build image. Retrain or resave serialized artifacts
with this version before deploying them.
Upload trained files to `models_artifacts/` using these exact names:
`rider_churn.joblib`, `rider_ltv.joblib`, `driver_eta.joblib`, and
`driver_acceptance.joblib`.

All endpoints expect JSON in the format:

```json
{
  "features": {
    "feature_name": 1.23
  }
}
```
