"""FastAPI service for the crop recommender (Module 6 deployment option).

Run:   uvicorn api:app --host 0.0.0.0 --port 8000
Docs:  http://localhost:8000/docs   (interactive Swagger UI)

Inputs are validated twice: by the Pydantic schema (types and hard limits, HTTP 422)
and by recommender.validate_input (shared with the Streamlit app). Each successful
prediction is appended to a JSON-lines log (inputs + top crop + confidence, no personal
data) that monitoring/drift_check.py can analyse.
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from recommender import HARD_LIMITS, load_model, recommend

BASE = Path(__file__).resolve().parent
LOG_PATH = Path(os.environ.get("PREDICTION_LOG", BASE / "logs" / "predictions.jsonl"))

model, meta = load_model()
app = FastAPI(title="Crop Recommendation API", version="1.0",
              description="Decision-support demo trained on a public Indian dataset. "
                          "Confirm recommendations with a local extension officer.")


def _f(name, desc):
    lo, hi = HARD_LIMITS[name]
    return Field(..., ge=lo, le=hi, description=f"{desc} (valid {lo}-{hi})")


class PlotReadings(BaseModel):
    N: float = _f("N", "Nitrogen ratio in soil")
    P: float = _f("P", "Phosphorus ratio in soil")
    K: float = _f("K", "Potassium ratio in soil")
    temperature: float = _f("temperature", "Temperature in °C")
    humidity: float = _f("humidity", "Relative humidity in %")
    ph: float = _f("ph", "Soil pH")
    rainfall: float = _f("rainfall", "Rainfall in mm")


class CropScore(BaseModel):
    crop: str
    confidence: float


class Recommendation(BaseModel):
    top: list[CropScore]
    warnings: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "model": meta["model_name"], "classes": len(meta["classes"])}


@app.get("/model-info")
def model_info():
    return {"model": meta["model_name"], "features": meta["features"],
            "training_ranges": meta["training_ranges"], "test_metrics": meta["test_metrics"],
            "sklearn_version_trained": meta["sklearn_version"]}


@app.post("/predict", response_model=Recommendation)
def predict(readings: PlotReadings):
    t0 = time.perf_counter()
    result = recommend(readings.model_dump(), model, meta)
    if not result["ok"]:  # defensive: schema already enforces the same limits
        return JSONResponse(status_code=422, content={"errors": result["errors"]})
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **readings.model_dump(),
                                 "top_crop": result["top"][0]["crop"],
                                 "confidence": round(result["top"][0]["confidence"], 4),
                                 "n_warnings": len(result["warnings"]),
                                 "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}) + "\n")
    except OSError:
        pass  # logging must never break a prediction
    return {"top": result["top"], "warnings": result["warnings"]}
