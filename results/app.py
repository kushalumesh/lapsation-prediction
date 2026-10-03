"""
Lapsation Risk Predictor
========================
Streamlit app: enter policyholder details, get a lapse probability
with a SHAP-based explanation of what drove it.

Run with:  streamlit run app.py
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

MODEL_PATH = Path("models/xgboost_model.pkl")

# --- palette -----------------------------------------------------------------
CANVAS = "#EFEDE7"
CARD = "#FFFFFF"
INK = "#1F1D1A"
MUTED = "#6E685E"
LINE = "#E3DFD5"
AMBER = "#F5C04E"
AMBER_DEEP = "#9A6B1C"
RISK_HIGH = "#C14840"
RISK_MID = "#9A6B1C"
RISK_LOW = "#3F7D5A"

READABLE = {
    "tenure_years": "policy tenure",
    "premium_loading_pct": "underwriting loading",
    "age_at_inception": "age at inception",
    "prior_claim_count": "prior claims",
    "premium_to_sum_insured": "premium relative to cover",
    "annual_premium": "annual premium",
    "sum_insured": "sum insured",
    "product_type_IP": "income protection cover",
    "product_type_TPD": "TPD cover",
    "product_type_Trauma": "trauma cover",
    "smoker_status_Y": "smoker",
    "smoker_status_Unknown": "smoker status not recorded",
    "gender_M": "gender",
}

st.set_page_config(page_title="Lapsation & Ceded Exposure", page_icon="◆", layout="wide")

def html(block: str) -> None:
    """Render raw HTML. st.html skips Markdown entirely, so indentation can't
    turn a stylesheet into a code block. Falls back for older Streamlit."""
    if hasattr(st, "html"):
        st.html(block)
    else:
        st.markdown(block, unsafe_allow_html=True)



html(f"""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
/* Background: stacked bands echoing a reinsurance layer tower - retention at the
   base, cession stacked above. Very low contrast so it never competes with data. */
.stApp {{
background-color: {CANVAS};
background-image:
repeating-linear-gradient(180deg,
rgba(31,29,26,0.022) 0px, rgba(31,29,26,0.022) 1px,
transparent 1px, transparent 84px),
repeating-linear-gradient(90deg,
rgba(31,29,26,0.014) 0px, rgba(31,29,26,0.014) 1px,
transparent 1px, transparent 84px);
}}
[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

html, body, [class*="st-"], .stMarkdown {{
font-family: 'Inter', -apple-system, sans-serif;
color: {INK};
}}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px; }}

[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] * {{ color: {INK}; }}
[data-testid="stSidebar"] h2 {{
font-family: 'Outfit', sans-serif; font-weight: 500;
font-size: 0.82rem; letter-spacing: 0.04em; text-transform: none;
color: {MUTED}; margin: 1.5rem 0 0.1rem; padding-bottom: 0.4rem;
border-bottom: 1px solid {LINE};
}}

.stButton > button {{
background: {INK}; color: {CANVAS};
border: none; border-radius: 10px;
padding: 0.7rem 1rem; font-weight: 500; font-size: 0.94rem;
}}
.stButton > button:hover {{ background: #000; color: {AMBER}; }}
.stButton > button:focus-visible {{ outline: 3px solid {AMBER}; outline-offset: 2px; }}

/* Masthead */
.masthead {{
background: {INK}; border-radius: 16px;
padding: 1.5rem 1.8rem; margin-bottom: 1rem;
display: flex; align-items: baseline; gap: 0.8rem; flex-wrap: wrap;
}}
.masthead .brand {{
font-family: 'Outfit', sans-serif; font-size: 1.45rem;
font-weight: 500; letter-spacing: -0.02em; color: {CANVAS};
}}
.masthead .brandsub {{ color: #A49C8E; font-size: 0.9rem; }}

.card {{
background: {CARD}; border-radius: 16px;
padding: 1.6rem 1.8rem; border: 1px solid {LINE}; margin-bottom: 1rem;
}}
.card.accent {{ border-top: 3px solid {AMBER}; }}

.score {{ font-family: 'Outfit', sans-serif; font-weight: 300; font-size: 5rem;
line-height: 0.92; letter-spacing: -0.045em; }}
.scorecap {{ color: {MUTED}; font-size: 0.9rem; margin-top: 0.5rem; }}
.pill {{ display: inline-block; border-radius: 6px; padding: 0.32rem 0.8rem;
font-size: 0.88rem; font-weight: 600; }}
.cardhead {{ font-family: 'Outfit', sans-serif; font-size: 1.05rem;
font-weight: 500; letter-spacing: -0.01em; }}
.eyebrow {{ color: {MUTED}; font-size: 0.78rem; letter-spacing: 0.04em; margin-bottom: 0.5rem; }}
.note {{ color: {MUTED}; font-size: 0.86rem; line-height: 1.6; }}
.readout {{ font-size: 1rem; line-height: 1.65; }}
.bignum {{ font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 500; }}

.track {{ height: 10px; border-radius: 5px; background: #E8E3D9; overflow: hidden;
margin: 0.8rem 0 0.4rem; }}
.fill {{ height: 100%; border-radius: 5px; }}
.ticks {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: 0.76rem; }}

/* Risk tower: retention block with ceded layers stacked above it */
.tower {{ display: flex; flex-direction: column; width: 100%; margin-top: 0.8rem;
border-radius: 8px; overflow: hidden; border: 1px solid {LINE}; }}
.layer {{ display: flex; justify-content: space-between; align-items: center;
padding: 0.6rem 0.85rem; font-size: 0.85rem; }}
.layer-ceded {{ background: {AMBER}; color: {INK}; }}
.layer-retained {{ background: #EDE9E0; color: {INK}; }}
.layer b {{ font-weight: 600; }}

/* Stat tiles */
.tiles {{ display: flex; gap: 0.7rem; flex-wrap: wrap; margin-bottom: 1rem; }}
.tile {{ flex: 1 1 0; min-width: 150px; background: {CARD}; border: 1px solid {LINE};
border-radius: 14px; padding: 1.1rem 1.2rem; }}
.tile.dark {{ background: {INK}; border-color: {INK}; }}
.tile.dark .tilenum, .tile.dark .tilecap {{ color: {CANVAS}; }}
.tile.dark .tilecap {{ color: #A49C8E; }}
.tile.amber {{ background: {AMBER}; border-color: {AMBER}; }}
.tilenum {{ font-family: 'Outfit', sans-serif; font-size: 1.9rem; font-weight: 500;
line-height: 1; letter-spacing: -0.03em; }}
.tilecap {{ color: {MUTED}; font-size: 0.78rem; margin-top: 0.45rem; line-height: 1.4; }}
.tile.amber .tilecap {{ color: #5E4A14; }}

/* Hero split */
.hero {{ display: flex; gap: 1rem; flex-wrap: wrap; align-items: stretch; }}
.hero > div {{ flex: 1 1 320px; }}
.steps {{ counter-reset: s; margin: 0.9rem 0 0; padding: 0; list-style: none; }}
.steps li {{ counter-increment: s; position: relative; padding-left: 2rem;
margin-bottom: 0.85rem; font-size: 0.92rem; line-height: 1.5; }}
.steps li::before {{ content: counter(s); position: absolute; left: 0; top: -1px;
width: 1.4rem; height: 1.4rem; border-radius: 5px; background: {INK}; color: {CANVAS};
font-size: 0.75rem; font-weight: 600; display: flex; align-items: center; justify-content: center; }}
</style>
""")


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


def explain(model, row):
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    transformed = prep.transform(row)
    names = [f.split("__", 1)[1] for f in prep.get_feature_names_out()]

    values = shap.TreeExplainer(clf).shap_values(transformed)[0]
    proba = float(model.predict_proba(row)[0, 1])

    contrib = pd.DataFrame({"feature": names, "shap": values})
    contrib["label"] = contrib.feature.map(lambda f: READABLE.get(f, f))
    return proba, contrib.reindex(contrib.shap.abs().sort_values(ascending=False).index)


def risk_band(proba, threshold):
    if proba >= threshold:
        return "High risk", RISK_HIGH, "Prioritise for retention contact."
    if proba >= threshold * 0.6:
        return "Moderate risk", RISK_MID, "Include in the next retention campaign."
    return "Low risk", RISK_LOW, "No action needed."


def driver_chart(contrib):
    top = contrib.head(6).sort_values("shap")
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    colors = [RISK_HIGH if v > 0 else "#8C9AA6" for v in top.shap]
    ax.barh(top.label, top.shap, color=colors, height=0.6)
    ax.axvline(0, color=INK, lw=1.0)

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, labelsize=10.5, colors=INK)
    ax.set_xticks([])
    ax.grid(False)

    span = max(abs(top.shap.min()), abs(top.shap.max())) * 1.12
    ax.set_xlim(-span, span)
    ax.text(span, -0.9, "raises risk", ha="right", va="center",
            fontsize=9, color=RISK_HIGH)
    ax.text(-span, -0.9, "lowers risk", ha="left", va="center",
            fontsize=9, color="#6E685E")
    fig.tight_layout()
    return fig


# --- page --------------------------------------------------------------------
html('<div class="masthead">'
'<span class="brand">Lapsation &amp; Ceded Exposure</span>'
'<span class="brandsub">Life insurance retention screening, priced through the reinsurance treaty</span>'
'</div>')

if not MODEL_PATH.exists():
    st.error(
        "No trained model found. Run `python generate_data.py`, then notebooks "
        "01 through 03 to train and save the model."
    )
    st.stop()

artefacts = load_model()
model, threshold = artefacts["model"], artefacts["threshold"]

with st.sidebar:
    st.markdown("## Policyholder")
    age = st.slider("Age at inception", 18, 75, 35)
    gender = st.selectbox("Gender", ["M", "F"])
    smoker = st.selectbox("Smoker", ["N", "Y"])
    product = st.selectbox("Product", ["Death", "TPD", "IP", "Trauma"])

    st.markdown("## Policy")
    sum_insured = st.number_input("Sum insured (AUD)", 50_000, 5_000_000, 500_000, step=50_000)
    premium = st.number_input("Annual premium (AUD)", 100.0, 50_000.0, 1_200.0, step=100.0)
    loading = st.slider("Underwriting loading (%)", 0.0, 50.0, 0.0, step=5.0)
    tenure_months = st.slider("Tenure (months)", 1, 300, 18)
    claims = st.number_input("Prior claims", 0, 10, 0)

    st.markdown("## Reinsurance treaty")
    treaty = st.selectbox("Treaty type", ["Quota Share", "Surplus"])
    if treaty == "Quota Share":
        cession = st.select_slider("Cession", [25.0, 40.0, 50.0], value=40.0,
                                   format_func=lambda v: f"{v:.0f}%")
    else:
        retention = 300_000
        cession = max(0.0, (sum_insured - retention) / sum_insured * 100)
        html(f'<div class="note">Retention AUD {retention:,} per life &rarr; '
             f'<b>{cession:.1f}%</b> of this policy is ceded.</div>')

    st.write("")
    predict = st.button("Score this policyholder", use_container_width=True)

if not predict:
    html(f"""
<div class="tiles">
<div class="tile dark">
<div class="tilenum">100,000</div>
<div class="tilecap">synthetic policyholders, calibrated to APRA 2024 industry statistics</div>
</div>
<div class="tile">
<div class="tilenum">8.3%</div>
<div class="tilecap">lapse each year &mdash; the rare event the model has to find</div>
</div>
<div class="tile">
<div class="tilenum">47%</div>
<div class="tilecap">of every premium dollar is ceded to a reinsurer</div>
</div>
<div class="tile amber">
<div class="tilenum">1.7&times;</div>
<div class="tilecap">better than chance at ranking who leaves next</div>
</div>
</div>

<div class="hero">
<div class="card">
<div class="eyebrow">WHAT THIS DOES</div>
<div class="cardhead">Find the policyholders worth calling</div>
<ol class="steps">
<li>Score a policyholder on nine behavioural and policy features.</li>
<li>Read which factors pushed that score up or down, from SHAP values.</li>
<li>Price the risk through the treaty &mdash; what their leaving costs the reinsurance programme.</li>
</ol>
<div class="note" style="margin-top:1.2rem;">
Set the details in the panel on the left, then select
<b style="color:{INK};">Score this policyholder</b>.
</div>
</div>

<div class="card accent">
<div class="eyebrow">HOW IT IS BUILT</div>
<div class="cardhead">XGBoost, with an honest baseline</div>
<div class="note" style="margin-top:0.8rem;">
Only 8.3% of policies lapse, so a model that predicts "nobody leaves" scores 92% accuracy
and catches no one. Accuracy is the wrong measure. This model is judged on AUC-ROC and
precision&ndash;recall, and compared against a logistic regression baseline rather than
against nothing.
</div>
<div class="tower" style="margin-top:1.1rem;">
<div class="layer layer-ceded"><span>XGBoost</span><span><b>0.144 PR-AUC</b></span></div>
<div class="layer layer-retained"><span>Logistic regression</span><span><b>0.143</b></span></div>
<div class="layer layer-retained" style="background:#F5F2EA;">
<span style="color:{MUTED};">Random guessing</span><span style="color:{MUTED};"><b>0.084</b></span></div>
</div>
<div class="note" style="margin-top:0.9rem;">
The gain over the baseline is small, which is the real finding: these relationships are
mostly linear, and the simpler model would be a defensible choice in production.
</div>
</div>
</div>
""")
    st.stop()

row = build_row(age, gender, smoker, product, sum_insured, premium, loading, claims, tenure_months)
proba, contrib = explain(model, row)
band, band_color, action = risk_band(proba, threshold)

ceded_premium = premium * cession / 100
ceded_cover = sum_insured * cession / 100
premium_at_risk = ceded_premium * proba

# Summary strip, so the three numbers that matter are visible before anything else
html(f"""
<div class="tiles">
<div class="tile" style="border-left:4px solid {band_color};">
<div class="tilenum" style="color:{band_color};">{proba:.0%}</div>
<div class="tilecap">chance of lapsing &mdash; {band.lower()}</div>
</div>
<div class="tile">
<div class="tilenum">{cession:.0f}%</div>
<div class="tilecap">ceded under the {treaty.lower()} treaty</div>
</div>
<div class="tile">
<div class="tilenum">${ceded_premium:,.0f}</div>
<div class="tilecap">ceded premium a year</div>
</div>
<div class="tile dark">
<div class="tilenum">${premium_at_risk:,.0f}</div>
<div class="tilecap">ceded premium at risk if they leave</div>
</div>
</div>
""")

left, right = st.columns([1, 1.45], gap="medium")

with left:
    html(f"""<div class="card">
<div class="score">{proba * 100:.0f}<span style="font-size:2.2rem">%</span></div>
<div class="scorecap">chance of lapsing</div>
<div class="track">
<div class="fill" style="width:{max(proba * 100, 2):.1f}%; background:{band_color};"></div>
</div>
<div class="ticks"><span>0%</span><span>flagged at {threshold:.0%}</span><span>100%</span></div>
<div style="margin-top:1.5rem;">
<span class="pill" style="background:{band_color}1A; color:{band_color};">{band}</span>
</div>
<div class="note" style="margin-top:0.9rem;">{action}</div>
</div>""")

with right:
    html(f"""<div class="card" style="margin-bottom:0.9rem;">
<div class="cardhead">Exposure under the {treaty.lower()} treaty</div>
<div style="display:flex; gap:2.6rem; margin-top:1rem; flex-wrap:wrap;">
<div>
<div class="bignum">
${ceded_premium:,.0f}</div>
<div class="note">ceded premium a year ({cession:.1f}% of ${premium:,.0f})</div>
</div>
<div>
<div class="bignum">
${ceded_cover:,.0f}</div>
<div class="note">ceded cover</div>
</div>
<div>
<div class="bignum" style="color:{band_color};">
${premium_at_risk:,.0f}</div>
<div class="note">ceded premium at risk</div>
</div>
</div>
<div class="note" style="margin-top:1.1rem; max-width:62ch;">
Ceded premium at risk is the annual ceded premium weighted by the lapse probability.
It is what this policyholder leaving would cost the reinsurance programme, not just the direct book.
</div>
</div>""")

    raising = [r.label for r in contrib.head(3).itertuples() if r.shap > 0]
    lowering = [r.label for r in contrib.head(3).itertuples() if r.shap < 0]

    sentence = ""
    if raising:
        sentence += f"Pushing risk up: <b>{', '.join(raising)}</b>. "
    if lowering:
        sentence += f"Holding risk down: <b>{', '.join(lowering)}</b>."

    html(f"""<div class="card" style="padding-bottom:1.1rem;">
<div class="cardhead">What drove this score</div>
<div class="readout" style="margin-top:0.6rem;">{sentence}</div>
</div>""")
    st.pyplot(driver_chart(contrib), use_container_width=True)
    html('<div class="note">Bars show each factor\'s contribution to this '
        'prediction. Right of the line raises risk, left lowers it.</div>')

with st.expander("All factor contributions"):
    st.dataframe(
        contrib[["label", "shap"]]
        .rename(columns={"label": "Factor", "shap": "Contribution"})
        .round(4),
        use_container_width=True, hide_index=True,
    )
