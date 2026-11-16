"""Experiment tracking with MLflow (Module 6, Task 2).

Re-runs the candidate models from the Module 5 notebook on the SAME split
(70/15/15, seed 42) and records parameters, metrics and the final model in a
local MLflow store (SQLite file ``mlflow.db`` + ``mlruns/`` artifacts).
A plain CSV copy is also written so reviewers without MLflow can read it.

Run:      python track_experiments.py
View UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import json
import time
import warnings
from pathlib import Path

import mlflow
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from recommender import FEATURES

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
RS = 42

df = pd.read_csv(BASE / "data" / "Crop_recommendation.csv")
X, y = df[FEATURES], df["label"]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RS)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RS)
cv5 = StratifiedKFold(5, shuffle=True, random_state=RS)

metrics_json = json.loads((BASE / "outputs" / "metrics.json").read_text())
rf_tuned = {k.replace("clf__", ""): v for k, v in metrics_json["rf_best_params"].items()}

def pipe(clf, scale):
    steps = ([("scaler", StandardScaler())] if scale else []) + [("clf", clf)]
    return Pipeline(steps)

CANDIDATES = {
    "KNN (baseline)": (pipe(KNeighborsClassifier(), True), {"n_neighbors": 5, "scaling": True}),
    "Decision Tree (baseline)": (pipe(DecisionTreeClassifier(random_state=RS), False), {"max_depth": None}),
    "Random Forest (default)": (pipe(RandomForestClassifier(random_state=RS), False), {"n_estimators": 100}),
    "Random Forest (tuned)": (pipe(RandomForestClassifier(random_state=RS, **rf_tuned), False), rf_tuned),
    "SVM (default, calibrated)": (pipe(CalibratedClassifierCV(SVC(random_state=RS), ensemble=False), True), {"C": 1.0, "kernel": "rbf"}),
    "Gaussian Naive Bayes": (pipe(GaussianNB(), False), {}),
}
FINAL = "Random Forest (default)"  # chosen in notebook Task 6 by the pre-declared selection rule


def scores(model, X_, y_):
    pred, proba = model.predict(X_), model.predict_proba(X_)
    rec = recall_score(y_, pred, labels=model.classes_, average=None, zero_division=0)
    return {"macro_f1": f1_score(y_, pred, average="macro"), "accuracy": accuracy_score(y_, pred),
            "min_class_recall": float(rec.min()),
            "top3_accuracy": top_k_accuracy_score(y_, proba, k=3, labels=model.classes_)}


mlflow.set_tracking_uri(f"sqlite:///{BASE / 'mlflow.db'}")
mlflow.set_experiment("crop-recommendation")
rows = []
for name, (model, params) in CANDIDATES.items():
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"group": "Group 6", "data": "Kaggle Crop Recommendation (real, not simulated)",
                         "is_final_model": str(name == FINAL)})
        mlflow.log_params({"model": name, "split": "70/15/15 stratified", "seed": RS,
                           "sklearn": sklearn.__version__, **{f"hp_{k}": v for k, v in params.items()}})
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        mlflow.log_metric("train_seconds", time.perf_counter() - t0)
        val = scores(model, X_val, y_val)
        cv = cross_val_score(model, X_train, y_train, cv=cv5, scoring="f1_macro", n_jobs=-1)
        mlflow.log_metrics({f"val_{k}": v for k, v in val.items()})
        mlflow.log_metrics({"cv_macro_f1_mean": cv.mean(), "cv_macro_f1_std": cv.std()})
        row = {"model": name, **{f"val_{k}": round(v, 4) for k, v in val.items()},
               "cv_macro_f1": round(cv.mean(), 4), "cv_std": round(cv.std(), 4)}
        if name in (FINAL, "KNN (baseline)"):  # test set only for the final model and the reference baseline
            test = scores(model, X_test, y_test)
            mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
            row.update({f"test_{k}": round(v, 4) for k, v in test.items()})
        if name == FINAL:
            mlflow.log_artifact(str(BASE / "models" / "crop_model.joblib"))  # our own trusted file
            mlflow.log_artifact(str(BASE / "models" / "model_meta.json"))
            mlflow.log_artifact(str(BASE / "outputs" / "metrics.json"))
        rows.append(row)
        print(f"logged: {name:28s} val macro F1 {val['macro_f1']:.4f}  cv {cv.mean():.4f}")

out = pd.DataFrame(rows)
out.to_csv(BASE / "outputs" / "experiment_log.csv", index=False)
print("\nSaved outputs/experiment_log.csv and MLflow store (mlflow.db, mlruns/).")
