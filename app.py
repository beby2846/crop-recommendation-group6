"""Crop Recommendation – Streamlit prototype (Group 6, FUSE AI-201, Module 5).

Run with:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from recommender import FEATURES, HARD_LIMITS, load_model, recommend

st.set_page_config(page_title="Crop Recommendation", page_icon="🌱", layout="centered")

LABELS = {
    "N": ("Nitrogen (N)", "ratio of N in soil"),
    "P": ("Phosphorus (P)", "ratio of P in soil"),
    "K": ("Potassium (K)", "ratio of K in soil"),
    "temperature": ("Temperature", "°C"),
    "humidity": ("Relative humidity", "%"),
    "ph": ("Soil pH", "0–14"),
    "rainfall": ("Rainfall", "mm"),
}
DATA_PATH = Path(__file__).resolve().parent / "data" / "Crop_recommendation.csv"


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_presets():
    """Median readings per crop, used only for the 'example plot' shortcut."""
    if not DATA_PATH.exists():
        return {}
    df = pd.read_csv(DATA_PATH)
    return df.groupby("label")[FEATURES].median().round(2).to_dict(orient="index")


def apply_preset():
    choice = st.session_state["preset"]
    presets = get_presets()
    if choice in presets:
        for f in FEATURES:
            st.session_state[f"in_{f}"] = float(presets[choice][f])


try:
    model, meta = get_model()
except Exception as exc:  # model file missing or incompatible scikit-learn version
    st.error("The trained model could not be loaded. Run the notebook first to create "
             f"`models/crop_model.joblib`.\n\nDetails: {exc}")
    st.stop()

ranges = meta["training_ranges"]
defaults = {"N": 80.0, "P": 45.0, "K": 40.0, "temperature": 25.0, "humidity": 70.0, "ph": 6.5, "rainfall": 100.0}
for f in FEATURES:
    st.session_state.setdefault(f"in_{f}", defaults[f])

st.title("🌱 Crop Recommendation")
st.caption("Enter your plot's soil and climate readings to get a ranked crop suggestion.")
st.warning(
    "**Decision-support demo, not a validated advisory service.** The model was trained on a public dataset "
    "reflecting Indian soil and climate conditions. Recommendations for other regions may be unreliable. "
    "Always confirm with a local agricultural extension officer before making planting decisions."
)

presets = get_presets()
if presets:
    st.selectbox("Start from an example plot (optional)", ["— enter my own values —"] + sorted(presets),
                 key="preset", on_change=apply_preset,
                 help="Fills the form with the typical readings of a crop so you can try the tool quickly.")

with st.form("plot_form"):
    c1, c2 = st.columns(2)
    for i, f in enumerate(FEATURES):
        lo, hi = HARD_LIMITS[f]
        name, unit = LABELS[f]
        with (c1 if i % 2 == 0 else c2):
            st.number_input(f"{name} — {unit}", min_value=float(lo), max_value=float(hi),
                            step=1.0 if f in ("N", "P", "K", "rainfall") else 0.1,
                            key=f"in_{f}",
                            help=f"Range seen in training data: {ranges[f][0]:.1f} to {ranges[f][1]:.1f}")
    submitted = st.form_submit_button("Recommend a crop", type="primary", use_container_width=True)

if submitted:
    values = {f: st.session_state[f"in_{f}"] for f in FEATURES}
    result = recommend(values, model, meta)

    if not result["ok"]:
        for e in result["errors"]:
            st.error(e)
    else:
        for w in result["warnings"]:
            st.warning(w)
        best = result["top"][0]
        st.subheader("Recommended crop")
        st.success(f"## {best['crop'].title()}\nConfidence: **{best['confidence']:.0%}**")

        st.markdown("**Next best alternatives**")
        for alt in result["top"][1:]:
            st.write(f"{alt['crop'].title()} — {alt['confidence']:.0%}")
            st.progress(min(max(alt["confidence"], 0.0), 1.0))

        with st.expander("Why this recommendation? (overall importance of each reading)"):
            imp = pd.Series(meta["feature_importance"]).sort_values()
            imp.index = [LABELS[i][0] for i in imp.index]
            st.bar_chart(imp, horizontal=True)
            st.caption("Global importance across all test plots (permutation importance). It shows which "
                       "readings the model relies on overall, not a per-plot explanation.")

with st.expander("About this tool"):
    tm = meta["test_metrics"]
    st.markdown(
        f"- **Model:** {meta['model_name']} (22 crops, 7 inputs)\n"
        f"- **Held-out test performance:** macro F1 {tm['macro_f1']:.3f}, top-3 accuracy {tm['top3_accuracy']:.1%}. "
        "This is measured on a clean benchmark; real farms will be noisier.\n"
        "- **Not considered:** market prices, irrigation access, seed availability, farmer preference.\n"
        "- Group 6 · FUSE AI-201 YEEAP Capstone · Module 5 prototype."
    )
