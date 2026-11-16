"""Data-drift and prediction-health check (Module 6, Task 3 – monitoring demo).

Compares NEW readings (a CSV with the 7 feature columns, optionally a 'label' column
if the true crop later becomes known) against the training reference and reports:
  * Population Stability Index (PSI) and a two-sample KS test per feature
  * share of inputs outside the range seen in training
  * mean model confidence and share of low-confidence predictions
  * accuracy / macro-F1 if labels are present
and turns them into a plain recommendation (OK / WATCH / RETRAIN REVIEW).

Usage:  python monitoring/drift_check.py new_data.csv [--out outputs/drift_report.csv]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recommender import FEATURES, load_model  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
PSI_WATCH, PSI_ALERT = 0.10, 0.25          # common rule-of-thumb thresholds
OUT_OF_RANGE_ALERT = 0.05                  # >5% of inputs outside training range
LOW_CONF = 0.5


def reference_data():
    df = pd.read_csv(BASE / "data" / "Crop_recommendation.csv")
    train, _ = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    return train


def psi(ref, new, bins=10):
    """Population Stability Index using reference-quantile bins."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    n = np.histogram(new, edges)[0] / len(new)
    r, n = np.clip(r, 1e-4, None), np.clip(n, 1e-4, None)
    return float(np.sum((n - r) * np.log(n / r)))


def check(new: pd.DataFrame):
    ref = reference_data()
    model, meta = load_model()
    rows = []
    for f in FEATURES:
        p = psi(ref[f].values, new[f].values)
        ks = ks_2samp(ref[f].values, new[f].values)
        lo, hi = meta["training_ranges"][f]
        rows.append({"feature": f, "psi": round(p, 3), "ks_stat": round(ks.statistic, 3),
                     "ks_pvalue": float(f"{ks.pvalue:.3g}"),
                     "ref_mean": round(ref[f].mean(), 2), "new_mean": round(new[f].mean(), 2),
                     "pct_outside_train_range": round(100 * ((new[f] < lo) | (new[f] > hi)).mean(), 1),
                     "status": "ALERT" if p >= PSI_ALERT else "WATCH" if p >= PSI_WATCH else "ok"})
    report = pd.DataFrame(rows)

    proba = model.predict_proba(new[FEATURES])
    pred = model.classes_[proba.argmax(1)]
    conf = proba.max(1)
    out_any = np.any([(new[f] < meta["training_ranges"][f][0]) | (new[f] > meta["training_ranges"][f][1])
                      for f in FEATURES], axis=0)
    summary = {"n_rows": len(new), "mean_confidence": round(float(conf.mean()), 3),
               "pct_low_confidence": round(100 * float((conf < LOW_CONF).mean()), 1),
               "pct_rows_any_feature_outside_range": round(100 * float(out_any.mean()), 1),
               "n_distinct_crops_predicted": int(len(set(pred)))}
    if "label" in new:
        summary["accuracy"] = round(float(accuracy_score(new["label"], pred)), 4)
        summary["macro_f1"] = round(float(f1_score(new["label"], pred, average="macro")), 4)

    n_alert = int((report.status == "ALERT").sum())
    n_watch = int((report.status == "WATCH").sum())
    if n_alert or summary["pct_rows_any_feature_outside_range"] > 100 * OUT_OF_RANGE_ALERT \
            or ("macro_f1" in summary and summary["macro_f1"] < 0.90):
        verdict = "RETRAIN REVIEW – significant drift or degraded performance: collect labelled data, re-evaluate, consider retraining."
    elif n_watch or summary["pct_low_confidence"] > 10:
        verdict = "WATCH – mild drift: increase monitoring frequency and review again next cycle."
    else:
        verdict = "OK – no meaningful drift detected."
    summary["verdict"] = verdict
    return report, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=None, help="write per-feature report to this CSV")
    a = ap.parse_args()
    data = pd.read_csv(a.csv)
    rep, summ = check(data)
    print(rep.to_string(index=False)); print()
    for k, v in summ.items():
        print(f"{k}: {v}")
    if a.out:
        rep.to_csv(a.out, index=False)
