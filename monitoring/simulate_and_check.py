"""Demonstration of the monitoring workflow on SIMULATED incoming data.

IMPORTANT: the 'new season' data below is SIMULATED from the held-out test rows. It is NOT real
field data and the results are NOT evidence of production performance. It only shows that the
monitoring code detects the kind of change it is designed to detect.

  scenario A "stable":   test rows + small measurement noise (5% of feature std)
  scenario B "drifted":  same rows with a simulated warmer, drier season
                         (temperature +5 °C, rainfall x0.6, humidity -12 points)
Labels are kept from the original rows (i.e. the assumption that crop requirements did not change).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drift_check import check, BASE  # noqa: E402
from recommender import FEATURES, HARD_LIMITS  # noqa: E402

df = pd.read_csv(BASE / "data" / "Crop_recommendation.csv")
_, tmp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
_, test = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=42)
test = test.reset_index(drop=True)
sd = df[FEATURES].std()
rng = np.random.default_rng(7)

def clip(d):
    for f in FEATURES:
        d[f] = d[f].clip(*HARD_LIMITS[f])
    return d

stable = test.copy()
stable[FEATURES] = stable[FEATURES] + rng.normal(0, 1, (len(stable), 7)) * sd.values * 0.05
drift = test.copy()
drift["temperature"] += 5
drift["rainfall"] *= 0.6
drift["humidity"] -= 12
stable, drift = clip(stable), clip(drift)

out = BASE / "outputs"
results = {}
for name, data in [("A_stable", stable), ("B_drifted", drift)]:
    data.to_csv(out / f"simulated_{name}.csv", index=False)
    rep, summ = check(data)
    rep.to_csv(out / f"drift_report_{name}.csv", index=False)
    results[name] = (rep, summ)
    print(f"=== Scenario {name} (SIMULATED) ===")
    print(rep[["feature", "psi", "ks_stat", "pct_outside_train_range", "status"]].to_string(index=False))
    for k, v in summ.items():
        print(f"  {k}: {v}")
    print()

# figure: PSI per feature for both scenarios
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(FEATURES)); w = 0.38
ax.bar(x - w/2, results["A_stable"][0].psi, w, label="A: stable (simulated)", color="#2f6f4f")
ax.bar(x + w/2, results["B_drifted"][0].psi, w, label="B: warmer/drier season (simulated)", color="#c46a1c")
ax.axhline(0.10, ls="--", c="grey", lw=1); ax.axhline(0.25, ls="--", c="red", lw=1)
ax.text(len(FEATURES) - 0.5, 0.105, "watch 0.10", ha="right", fontsize=8, color="grey")
ax.text(len(FEATURES) - 0.5, 0.255, "alert 0.25", ha="right", fontsize=8, color="red")
ax.set_xticks(x); ax.set_xticklabels(FEATURES); ax.set_ylabel("PSI (higher = more drift)")
ax.set_title("Monitoring demo on SIMULATED data: feature drift (PSI)")
ax.legend(); plt.tight_layout(); plt.savefig(out / "drift_demo.png", dpi=150)
