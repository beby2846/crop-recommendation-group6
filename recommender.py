"""Crop recommendation helpers shared by the notebook and the Streamlit app.

Keeping the feature-engineering function and the prediction logic in one
importable module means the saved model (joblib) can be loaded anywhere.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Hard limits: values outside these are physically impossible / invalid input.
HARD_LIMITS = {
    "N": (0, 500),
    "P": (0, 500),
    "K": (0, 500),
    "temperature": (-10, 60),
    "humidity": (0, 100),
    "ph": (0, 14),
    "rainfall": (0, 1000),
}

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "crop_model.joblib"
META_PATH = BASE_DIR / "models" / "model_meta.json"


def add_ratio_features(X):
    """Optional feature engineering used in the notebook experiment.

    Adds N:P, N:K and P:K ratios (with +1 smoothing to avoid divide-by-zero).
    Works on a DataFrame or a NumPy array with the 7 base columns in order.
    """
    arr = np.asarray(X, dtype=float)
    n, p, k = arr[:, 0], arr[:, 1], arr[:, 2]
    extra = np.column_stack([(n + 1) / (p + 1), (n + 1) / (k + 1), (p + 1) / (k + 1)])
    return np.hstack([arr, extra])


def load_model(model_path: Path = MODEL_PATH, meta_path: Path = META_PATH):
    model = joblib.load(model_path)
    meta = json.loads(Path(meta_path).read_text())
    return model, meta


def validate_input(values: dict, training_ranges: dict | None = None):
    """Validate one plot's readings.

    Returns (clean_values, errors, warnings).
    - errors: input cannot be used (missing, non-numeric, impossible value)
    - warnings: usable, but outside the range the model was trained on
    """
    errors, warnings, clean = [], [], {}
    for f in FEATURES:
        if f not in values or values[f] is None or values[f] == "":
            errors.append(f"Missing value for '{f}'.")
            continue
        try:
            v = float(values[f])
        except (TypeError, ValueError):
            errors.append(f"'{f}' must be a number (got {values[f]!r}).")
            continue
        if math.isnan(v) or math.isinf(v):
            errors.append(f"'{f}' must be a finite number.")
            continue
        lo, hi = HARD_LIMITS[f]
        if not (lo <= v <= hi):
            errors.append(f"'{f}'={v} is outside the valid range [{lo}, {hi}].")
            continue
        clean[f] = v
        if training_ranges and f in training_ranges:
            tlo, thi = training_ranges[f]
            if v < tlo or v > thi:
                warnings.append(
                    f"'{f}'={v} is outside the range seen in training data "
                    f"[{tlo:.1f}, {thi:.1f}]; the recommendation may be unreliable."
                )
    return clean, errors, warnings


def recommend(values: dict, model=None, meta=None, top_k: int = 3):
    """Return a dict with the ranked recommendation, warnings and errors."""
    if model is None or meta is None:
        model, meta = load_model()
    clean, errors, warnings = validate_input(values, meta.get("training_ranges"))
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "top": []}

    row = pd.DataFrame([[clean[f] for f in FEATURES]], columns=FEATURES)
    proba = model.predict_proba(row)[0]
    classes = model.classes_
    order = np.argsort(proba)[::-1][:top_k]
    top = [{"crop": str(classes[i]), "confidence": float(proba[i])} for i in order]

    if top[0]["confidence"] < meta.get("low_confidence_threshold", 0.5):
        warnings.append(
            "The model is not confident about this plot (top confidence is low). "
            "Consider the alternatives and confirm with a local extension officer."
        )
    return {"ok": True, "errors": [], "warnings": warnings, "top": top}
