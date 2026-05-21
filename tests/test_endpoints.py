from fastapi.testclient import TestClient

from src.main import app


def test_root() -> None:
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()


def test_health_reports_model_loaded() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


def test_predict_returns_prediction() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"symbol": "GGAL", "features": {"opening": 1.0, "offered": 3.0}},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["symbol"] == "GGAL"
        assert body["prediction"] == 2.0
        assert "model_version" in body


def test_predict_rejects_empty_symbol() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json={"symbol": "", "features": {}})
        assert response.status_code == 422
