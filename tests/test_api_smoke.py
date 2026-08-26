from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_verification_health_is_available():
    response = client.get("/api/v1/verification/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_openapi_lists_verification_endpoints():
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/verification/verify-driver" in paths
    assert "/api/v1/verification/detect-liveness" in paths
