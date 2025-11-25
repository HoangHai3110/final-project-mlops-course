import os
import pytest
from fastapi.testclient import TestClient

from scripts.service.app import app

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")


@pytest.fixture(scope="session")
def client():
    """
    Dùng context manager để FastAPI chạy startup event (load_model)
    trước khi chạy test.
    """
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"


def test_model_info(client):
    resp = client.get("/model_info")
    assert resp.status_code == 200
    data = resp.json()
    assert "tracking_uri" in data
    assert "model_uri" in data
    assert "model_loaded" in data
    # assert data["model_loaded"] is True  
    assert isinstance(data["model_loaded"], bool)


def test_predict_single(client):
    payload = {
        "Contract": "Month-to-month",
        "tenure": 5,
        "MonthlyCharges": 80.5,
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert "churn_probability" in data
    assert "churn_predicted" in data

    assert isinstance(data["churn_probability"], float)
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_predicted"] in (0, 1)


def test_predict_batch(client):
    payload = {
        "records": [
            {
                "Contract": "Two year",
                "tenure": 30,
                "MonthlyCharges": 55.0,
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "TechSupport": "Yes",
            },
            {
                "Contract": "Two year",
                "tenure": 40,
                "MonthlyCharges": 60.0,
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "TechSupport": "Yes",
            },
            {
                "Contract": "One year",
                "tenure": 18,
                "MonthlyCharges": 50.0,
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "TechSupport": "No",
            },
            {
                "Contract": "Month-to-month",
                "tenure": 3,
                "MonthlyCharges": 85.0,
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "TechSupport": "No",
            },
            {
                "Contract": "Month-to-month",
                "tenure": 1,
                "MonthlyCharges": 90.0,
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "TechSupport": "No",
            },
        ]
    }

    resp = client.post("/predict_batch", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert "predictions" in data
    preds = data["predictions"]

    assert isinstance(preds, list)
    assert len(preds) == 5

    for p in preds:
        assert "churn_probability" in p
        assert "churn_predicted" in p
        assert 0.0 <= p["churn_probability"] <= 1.0
        assert p["churn_predicted"] in (0, 1)
