from fastapi.testclient import TestClient

from wsbench.server.app import app


def test_health():
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert response.json()["version"] == "0.1.0"
