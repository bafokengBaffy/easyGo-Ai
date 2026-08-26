---
title: easygoAI
emoji: "??"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
---

# easygoAI

Docker Space for EasyGo ML inference.

## Endpoints
- `GET /health`
- `POST /predict/rider_churn`
- `POST /predict/rider_ltv`
- `POST /predict/driver_eta`
- `POST /predict/driver_acceptance`

All endpoints expect JSON in the format:

```json
{
  "features": {
    "feature_name": 1.23
  }
}
```
