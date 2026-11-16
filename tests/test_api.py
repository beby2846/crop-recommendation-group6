import pandas as pd
from fastapi.testclient import TestClient
from api import app
from recommender import FEATURES

client = TestClient(app)
DATA = pd.read_csv("data/Crop_recommendation.csv")
MAIZE = DATA[DATA.label == "maize"][FEATURES].median().to_dict()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200 and "test_metrics" in r.json()


def test_predict_valid():
    r = client.post("/predict", json=MAIZE)
    assert r.status_code == 200
    body = r.json()
    assert body["top"][0]["crop"] == "maize" and len(body["top"]) == 3


def test_predict_out_of_range_input_is_422():
    r = client.post("/predict", json={**MAIZE, "humidity": 120})
    assert r.status_code == 422


def test_predict_missing_field_is_422():
    body = {k: v for k, v in MAIZE.items() if k != "ph"}
    assert client.post("/predict", json=body).status_code == 422


def test_predict_wrong_type_is_422():
    assert client.post("/predict", json={**MAIZE, "N": "abc"}).status_code == 422


def test_predict_empty_body_is_422():
    assert client.post("/predict", json={}).status_code == 422


def test_unusual_but_valid_input_returns_warning():
    r = client.post("/predict", json={**MAIZE, "ph": 14})
    assert r.status_code == 200 and r.json()["warnings"]


def test_prediction_is_logged():
    import os, json
    path = os.environ["PREDICTION_LOG"]
    client.post("/predict", json=MAIZE)
    lines = open(path).read().strip().splitlines()
    rec = json.loads(lines[-1])
    assert rec["top_crop"] == "maize" and "ts" in rec and "N" in rec
