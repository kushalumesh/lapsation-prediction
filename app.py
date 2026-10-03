"""
Lapsation Risk Predictor
========================
Streamlit app: enter policyholder details, get a lapse probability
with a SHAP-based explanation of what drove it.

Run with:  streamlit run app.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

MODEL_PATH = Path("models/xgboost_model.pkl")

READABLE = {
    "tenure_years": "policy tenure",
    "premium_loading_pct": "underwriting loading",
    "age_at_inception": "age at inception",
    "prior_claim_count": "prior claims",
    "premium_to_sum_insured": "premium relative to cover",
    "annual_premium": "annual premium",
    "sum_insured": "sum insured",
    "product_type_IP": "income protection product",
    "product_type_TPD": "TPD product",
    "product_type_Trauma": "trauma product",
    "smoker_status_Y": "smoker status",
    "smoker_status_Unknown": "unknown smoker status",
    "gender_M": "gender",
}

st.set_page_config(page_title="Lapsation Risk Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_row(age, gender, smoker, product, sum_insured, premium, loading, claims, tenure_months):
    return pd.DataFrame([{
        "age_at_inception": age,
        "gender": gender,
        "smoker_status": smoker,
        "product_type": product,
        "sum_insured": sum_insured,
        "annual_premium": premium,
        "premium_loading_pct": loading,
        "prior_claim_count": claims,
        "premium_to_sum_insured": premium / sum_insured * 1000,
        "tenure_years": tenure_months / 12,
    }])


def explain(model, row, threshold):
    """Return (probability, dataframe of SHAP contributions)."""
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]

    transformed = prep.transform(row)
    names = [f.split("__", 1)[1] for f in prep.get_feature_names_out()]

    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(transformed)[0]

    proba = float(model.predict_proba(row)[0, 1])
    contrib = pd.DataFrame({"feature": names, "shap": values})
    contrib["label"] = contrib.feature.map(lambda f: READABLE.get(f, f))
    contrib = contrib.reindex(contrib.shap.abs().sort_values(ascending=False).index)
    return proba, contrib


def risk_band(proba, threshold):
    if proba >= threshold:
        return "High", "🔴"
    if proba >= threshold * 0.6:
        return "Moderate", "🟠"
    return "Low", "🟢"


st.title("Lapsation Risk Predictor")
st.caption(
    "Predicts the probability that a life insurance policyholder cancels cover early, "
    "and explains which factors drove the prediction."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Run the notebooks first:\n\n"
        "`python generate_data.py`, then `01_eda` → `02_baseline_model` → `03_xgboost_model`."
    )
    st.stop()

artefacts = load_model()
model = artefacts["model"]
threshold = artefacts["threshold"]

with st.sidebar:
    st.header("Policyholder details")

    age = st.slider("Age at policy inception", 18, 75, 35)
    gender = st.selectbox("Gender", ["M", "F"])
    smoker = st.selectbox("Smoker", ["N", "Y"])
    product = st.selectbox("Product type", ["Death", "TPD", "IP", "Trauma"])

    st.divider()

    sum_insured = st.number_input("Sum insured (AUD)", 50_000, 5_000_000, 500_000, step=50_000)
    premium = st.number_input("Annual premium (AUD)", 100.0, 50_000.0, 1_200.0, step=100.0)
    loading = st.slider("Underwriting loading (%)", 0.0, 50.0, 0.0, step=5.0)

    st.divider()

    tenure_months = st.slider("Policy tenure (months)", 1, 300, 18)
    claims = st.number_input("Prior claims", 0, 10, 0)

    predict = st.button("Predict", type="primary", use_container_width=True)

if predict:
    row = build_row(age, gender, smoker, product, sum_insured, premium, loading, claims, tenure_months)
    proba, contrib = explain(model, row, threshold)
    band, icon = risk_band(proba, threshold)

    left, right = st.columns([1, 2])

    with left:
        st.metric("Lapse probability", f"{proba:.1%}")
        st.subheader(f"{icon} {band} risk")
        st.caption(f"Flagged when probability ≥ {threshold:.0%} (threshold tuned on validation data)")

        if band == "High":
            st.warning("Recommend prioritised retention contact.")
        elif band == "Moderate":
            st.info("Consider including in the next retention campaign.")
        else:
            st.success("No action required.")

    with right:
        st.subheader("What drove this prediction")

        top = contrib.head(6).sort_values("shap")
        fig, ax = plt.subplots(figsize=(7, 3.6))
        colors = ["#C44E52" if v > 0 else "#4C72B0" for v in top.shap]
        ax.barh(top.label, top.shap, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("SHAP value  (right = increases risk, left = reduces risk)")
        ax.set_title("Top contributing factors", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

        raising = [r.label for r in contrib.head(3).itertuples() if r.shap > 0]
        lowering = [r.label for r in contrib.head(3).itertuples() if r.shap < 0]

        parts = [f"**{band} lapse risk ({proba:.1%}).**"]
        if raising:
            parts.append("Increasing risk: " + ", ".join(raising) + ".")
        if lowering:
            parts.append("Reducing risk: " + ", ".join(lowering) + ".")
        st.markdown(" ".join(parts))

    with st.expander("Full feature contributions"):
        st.dataframe(
            contrib[["label", "shap"]]
            .rename(columns={"label": "Feature", "shap": "SHAP value"})
            .round(4),
            use_container_width=True, hide_index=True,
        )
else:
    st.info("Enter policyholder details in the sidebar, then select **Predict**.")

    st.subheader("About this model")
    st.markdown(
        """
        - **Data** — 100,000 synthetic policyholders, parameterised from APRA 2024 Life Insurance Quarterly Statistics
        - **Model** — XGBoost, compared against a logistic regression baseline
        - **Imbalance** — only ~8.3% of policies lapse, so the model is evaluated on AUC-ROC and PR-AUC rather than accuracy, with class weighting and a tuned decision threshold
        - **Explainability** — SHAP values for every individual prediction
        - **Fairness** — performance checked across age bands
        """
    )
