import math
import pandas as pd
import pytest
from recommender import FEATURES, load_model, recommend, validate_input

model, meta = load_model()
DATA = pd.read_csv("data/Crop_recommendation.csv")
RICE = DATA[DATA.label == "rice"][FEATURES].median().to_dict()


def test_typical_rice_ranked_first():
    out = recommend(RICE, model, meta)
    assert out["ok"] and out["top"][0]["crop"] == "rice"


def test_returns_three_ranked_alternatives_summing_below_one():
    out = recommend(RICE, model, meta)
    conf = [t["confidence"] for t in out["top"]]
    assert len(conf) == 3 and conf == sorted(conf, reverse=True) and sum(conf) <= 1.0 + 1e-9


@pytest.mark.parametrize("field,bad", [
    ("humidity", 120), ("humidity", -1), ("ph", 15), ("rainfall", -10), ("N", -1),
    ("temperature", 80), ("K", float("nan")), ("N", float("inf")), ("P", "abc"), ("ph", ""), ("ph", None),
])
def test_invalid_values_are_rejected(field, bad):
    out = recommend({**RICE, field: bad}, model, meta)
    assert not out["ok"] and out["errors"] and out["top"] == []


def test_missing_field_is_rejected():
    vals = {k: v for k, v in RICE.items() if k != "ph"}
    assert not recommend(vals, model, meta)["ok"]


def test_numbers_as_text_are_accepted():
    out = recommend({**RICE, "N": "90"}, model, meta)
    assert out["ok"]


def test_out_of_training_range_gives_warning_not_error():
    out = recommend({**RICE, "ph": 14}, model, meta)
    assert out["ok"] and any("outside the range" in w for w in out["warnings"])


def test_all_zero_nutrients_is_valid():
    out = recommend({**RICE, "N": 0, "P": 0, "K": 0}, model, meta)
    assert out["ok"]


def test_boundary_values_accepted():
    clean, errors, _ = validate_input({"N": 0, "P": 0, "K": 0, "temperature": -10, "humidity": 100, "ph": 14, "rainfall": 1000})
    assert not errors and len(clean) == 7


def test_saved_model_accuracy_on_full_dataset_is_high():
    # sanity check on a trained model (in-sample, so it should be very high)
    acc = (model.predict(DATA[FEATURES]) == DATA.label).mean()
    assert acc > 0.97


def test_metadata_is_consistent():
    assert meta["features"] == FEATURES and len(meta["classes"]) == 22
    assert set(meta["classes"]) == set(model.classes_)
