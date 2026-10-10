"""
Lapsation & Ceded Exposure
==========================
Scores a policyholder's chance of cancelling, explains what drove it,
and prices the cost through their reinsurance treaty.

Run with:  streamlit run app.py
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

MODEL_PATH = Path("models/xgboost_model.pkl")

# --- palette: carried over from the portfolio site --------------------------
BG       = "#0D0D0F"
CARD     = "#1C1C21"
LINE     = "#2A2A30"
TEXT     = "#F5F3EE"
BODY     = "#C9C8D1"
MUTED    = "#9A9AA3"
CORAL    = "#FF5A45"
BLUE     = "#4C8DFF"
TEAL     = "#2FD1A6"
YELLOW   = "#FFC93C"

RISK_HIGH = CORAL
RISK_MID  = YELLOW
RISK_LOW  = TEAL

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
    """Raw HTML with no Markdown processing, so indentation can't break it."""
    if hasattr(st, "html"):
        st.html(block)
    else:
        st.markdown(block, unsafe_allow_html=True)


html(f"""
<link href="https://fonts.googleapis.com/css2?family=Archivo+Black&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ---------------------------------------------------------------
   Animated backdrop. Streamlit strips <script>, so the motion is
   pure CSS: slow-drifting colour fields plus a travelling dot grid.
   Both are fixed, behind everything, and ignore pointer events.
   ---------------------------------------------------------------- */
.stApp {{ background: {BG}; }}

.stApp::before {{
content: ""; position: fixed; inset: -25%; z-index: 0; pointer-events: none;
background:
radial-gradient(38% 38% at 18% 22%, rgba(255,90,69,0.20), transparent 62%),
radial-gradient(34% 34% at 80% 18%, rgba(76,141,255,0.16), transparent 62%),
radial-gradient(40% 40% at 68% 82%, rgba(47,209,166,0.14), transparent 62%),
radial-gradient(28% 28% at 32% 88%, rgba(255,201,60,0.10), transparent 62%);
filter: blur(14px);
animation: drift 30s ease-in-out infinite alternate;
}}

.stApp::after {{
content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
background-image: radial-gradient(circle, rgba(201,200,209,0.10) 1px, transparent 1px);
background-size: 34px 34px;
animation: grid 38s linear infinite;
}}

@keyframes drift {{
0%   {{ transform: translate3d(0,0,0) scale(1); }}
50%  {{ transform: translate3d(2.5%, -2%, 0) scale(1.07); }}
100% {{ transform: translate3d(-2%, 2.5%, 0) scale(1.03); }}
}}
@keyframes grid {{ to {{ background-position: 34px 34px; }} }}

@media (prefers-reduced-motion: reduce) {{
.stApp::before, .stApp::after {{ animation: none; }}
}}

/* keep all real content above the backdrop */
.block-container, [data-testid="stSidebar"] {{ position: relative; z-index: 1; }}
[data-testid="stHeader"] {{ background: transparent; z-index: 2; }}
#MainMenu, footer {{ visibility: hidden; }}

/* ---- typography ---- */
html, body, .stMarkdown, .stApp {{
font-family: 'IBM Plex Sans', system-ui, sans-serif;
color: {TEXT};
}}
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded,
span[data-testid*="Icon"] {{
font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}}
.block-container {{ padding-top: 2rem; padding-bottom: 4rem; max-width: 1240px; }}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {{
background: rgba(20,20,23,0.92);
border-right: 1px solid {LINE};
backdrop-filter: blur(8px);
}}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] span, [data-testid="stSidebar"] div {{ color: {BODY}; }}
[data-testid="stSidebar"] h2 {{
font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
text-transform: uppercase; color: {CORAL};
margin: 1.6rem 0 0.3rem; padding-bottom: 0.5rem;
border-bottom: 1px solid {LINE};
}}

/* ---- primary action ---- */
.stButton > button {{
background: {CORAL}; border: none; border-radius: 999px;
padding: 0.8rem 1rem; font-weight: 700; font-size: 0.95rem;
transition: background 160ms ease, transform 160ms ease;
}}
.stButton > button, .stButton > button *,
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stButton > button * {{ color: {BG} !important; }}
.stButton > button:hover {{ background: #E64B37; transform: translateY(-2px); }}
.stButton > button:focus-visible {{ outline: 3px solid {YELLOW}; outline-offset: 3px; }}

/* ---- masthead ---- */
.masthead {{ margin-bottom: 1.6rem; }}
.wordmark {{
font-family: 'Archivo Black', sans-serif; font-size: 1.5rem;
letter-spacing: -0.01em; color: {TEXT};
}}
.wordmark .dot {{ color: {CORAL}; }}
.eyebrow {{
font-size: 0.75rem; font-weight: 600; letter-spacing: 0.12em;
text-transform: uppercase; color: {CORAL}; margin-bottom: 0.55rem;
}}
.display {{
font-family: 'Archivo Black', sans-serif; font-size: 2.6rem;
line-height: 1.04; color: {TEXT}; margin: 0.1rem 0;
}}
.display.ghost {{ color: transparent; -webkit-text-stroke: 1.6px {CORAL}; }}
.lede {{ color: {MUTED}; font-size: 1rem; line-height: 1.65; max-width: 62ch; margin-top: 0.9rem; }}

/* ---- cards ---- */
.card {{
background: rgba(28,28,33,0.86); border: 1px solid {LINE};
border-radius: 10px; padding: 1.7rem 1.8rem; margin-bottom: 1rem;
backdrop-filter: blur(6px);
transition: transform 180ms ease, box-shadow 180ms ease;
}}
.card:hover {{ transform: translateY(-3px); box-shadow: 0 14px 28px rgba(0,0,0,0.38); }}
.cardhead {{ font-size: 1.15rem; font-weight: 700; color: {TEXT}; }}
.note {{ color: {MUTED}; font-size: 0.88rem; line-height: 1.65; }}
.readout {{ color: {BODY}; font-size: 1rem; line-height: 1.7; }}
.readout b {{ color: {TEXT}; font-weight: 700; }}

/* ---- stat tiles ---- */
.tiles {{ display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 1rem; }}
.tile {{
flex: 1 1 0; min-width: 165px;
background: rgba(28,28,33,0.86); border: 1px solid {LINE};
border-top: 3px solid {LINE}; border-radius: 10px; padding: 1.2rem 1.3rem;
backdrop-filter: blur(6px);
transition: transform 180ms ease;
}}
.tile:hover {{ transform: translateY(-3px); }}
.tilenum {{
font-family: 'Archivo Black', sans-serif; font-size: 1.9rem;
line-height: 1; letter-spacing: -0.02em; color: {TEXT};
}}
.tilecap {{ color: {MUTED}; font-size: 0.78rem; line-height: 1.5; margin-top: 0.5rem; }}

/* ---- the score ---- */
.score {{
font-family: 'Archivo Black', sans-serif; font-size: 5.6rem;
line-height: 0.9; letter-spacing: -0.03em;
}}
.score .pc {{ font-size: 2.2rem; color: transparent; -webkit-text-stroke: 1.6px currentColor; }}
.scorecap {{ color: {MUTED}; font-size: 0.9rem; margin-top: 0.7rem; }}
.pill {{
display: inline-block; border-radius: 999px; padding: 0.35rem 1rem;
font-size: 0.85rem; font-weight: 700;
}}

/* ---- meters ---- */
.track {{ height: 8px; border-radius: 999px; background: #232327; overflow: hidden; margin: 1rem 0 0.45rem; }}
.fill {{ height: 100%; border-radius: 999px; }}
.ticks {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: 0.75rem; }}

/* ---- risk tower: ceded stacked above retained ---- */
.tower {{ display: flex; flex-direction: column; border-radius: 8px; overflow: hidden;
border: 1px solid {LINE}; margin-top: 0.9rem; }}
.layer {{ display: flex; justify-content: space-between; align-items: center;
padding: 0.7rem 0.95rem; font-size: 0.86rem; }}
.layer-ceded {{ background: {CORAL}; color: {BG}; font-weight: 600; }}
.layer-retained {{ background: #232327; color: {BODY}; }}
.layer b {{ font-weight: 700; }}

/* ---- step list ---- */
.steps {{ counter-reset: s; margin: 1rem 0 0; padding: 0; list-style: none; }}
.steps li {{ counter-increment: s; position: relative; padding-left: 2.1rem;
margin-bottom: 0.9rem; font-size: 0.94rem; line-height: 1.6; color: {BODY}; }}
.steps li::before {{
content: counter(s); position: absolute; left: 0; top: 0;
width: 1.5rem; height: 1.5rem; border-radius: 50%;
border: 1.5px solid {CORAL}; color: {CORAL};
font-size: 0.72rem; font-weight: 700;
display: flex; align-items: center; justify-content: center;
}}
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
    """Horizontal SHAP contributions, styled for the dark canvas."""
    top = contrib.head(6).sort_values("shap")
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    colors = [CORAL if v > 0 else BLUE for v in top.shap]
    ax.barh(top.label, top.shap, color=colors, height=0.6)
    ax.axvline(0, color=MUTED, lw=1.0)

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, labelsize=11, colors=BODY)
    ax.set_xticks([])
    ax.grid(False)

    span = max(abs(top.shap.min()), abs(top.shap.max())) * 1.14
    ax.set_xlim(-span, span)
    ax.text(span, -0.95, "raises risk", ha="right", va="center", fontsize=9, color=CORAL)
    ax.text(-span, -0.95, "lowers risk", ha="left", va="center", fontsize=9, color=BLUE)
    fig.tight_layout()
    return fig


# --- page -------------------------------------------------------------------
html(f"""
<div class="masthead">
<div class="wordmark">LAPSATION<span class="dot">.</span></div>
<div class="eyebrow" style="margin-top:1.3rem;">Retention screening</div>
<div class="display">Who leaves,</div>
<div class="display ghost">and what it costs.</div>
<div class="lede">Scores a life insurance policyholder's chance of cancelling early, shows
which factors drove the score, and prices the loss through the reinsurance treaty
the policy sits under.</div>
</div>
""")

if not MODEL_PATH.exists():
    st.error(
        "No trained model found. Run `python generate_data.py`, then notebooks "
        "01 through 03 to train and save the model."
    )
    st.stop()

artefacts = load_model()
model, threshold = artefacts["model"], artefacts["threshold"]

with st.sidebar:
    html(f'<div class="wordmark" style="font-size:1.1rem;">LAPSATION<span class="dot">.</span></div>')
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
             f'<b style="color:{CORAL};">{cession:.1f}%</b> of this policy is ceded.</div>')

    st.write("")
    predict = st.button("Score this policyholder", use_container_width=True)

if not predict:
    html(f"""
<div class="tiles">
<div class="tile" style="border-top-color:{CORAL};">
<div class="tilenum">100,000</div>
<div class="tilecap">synthetic policyholders, calibrated to APRA 2024 industry statistics</div>
</div>
<div class="tile" style="border-top-color:{BLUE};">
<div class="tilenum">8.3%</div>
<div class="tilecap">lapse each year &mdash; the rare event the model has to find</div>
</div>
<div class="tile" style="border-top-color:{TEAL};">
<div class="tilenum">47%</div>
<div class="tilecap">of every premium dollar is ceded to a reinsurer</div>
</div>
<div class="tile" style="border-top-color:{YELLOW};">
<div class="tilenum">1.7&times;</div>
<div class="tilecap">better than chance at ranking who leaves next</div>
</div>
</div>
""")

    left, right = st.columns(2, gap="medium")
    with left:
        html(f"""
<div class="card">
<div class="eyebrow">What this does</div>
<div class="cardhead">Find the policyholders worth calling</div>
<ol class="steps">
<li>Score a policyholder on nine behavioural and policy features.</li>
<li>Read which factors pushed that score up or down, from SHAP values.</li>
<li>Price the risk through the treaty &mdash; what their leaving costs the reinsurance programme.</li>
</ol>
<div class="note" style="margin-top:1.3rem;">
Set the details on the left, then select
<b style="color:{CORAL};">Score this policyholder</b>.
</div>
</div>
""")
    with right:
        html(f"""
<div class="card">
<div class="eyebrow">How it is built</div>
<div class="cardhead">XGBoost, with an honest baseline</div>
<div class="note" style="margin-top:0.8rem;">
Only 8.3% of policies lapse, so a model that predicts "nobody leaves" scores 92% accuracy
and catches no one. Accuracy is the wrong measure. This model is judged on AUC-ROC and
precision&ndash;recall, and compared against a logistic regression baseline rather than
against nothing.
</div>
<div class="tower">
<div class="layer layer-ceded"><span>XGBoost</span><span><b>0.144 PR-AUC</b></span></div>
<div class="layer layer-retained"><span>Logistic regression</span><span><b>0.143</b></span></div>
<div class="layer layer-retained" style="background:#1A1A1E;">
<span style="color:{MUTED};">Random guessing</span><span style="color:{MUTED};"><b>0.084</b></span></div>
</div>
<div class="note" style="margin-top:1rem;">
The gain over the baseline is small, which is the real finding: these relationships are
mostly linear, and the simpler model would be a defensible choice in production.
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

html(f"""
<div class="tiles">
<div class="tile" style="border-top-color:{band_color};">
<div class="tilenum" style="color:{band_color};">{proba:.0%}</div>
<div class="tilecap">chance of lapsing &mdash; {band.lower()}</div>
</div>
<div class="tile" style="border-top-color:{BLUE};">
<div class="tilenum">{cession:.0f}%</div>
<div class="tilecap">ceded under the {treaty.lower()} treaty</div>
</div>
<div class="tile" style="border-top-color:{TEAL};">
<div class="tilenum">${ceded_premium:,.0f}</div>
<div class="tilecap">ceded premium a year</div>
</div>
<div class="tile" style="border-top-color:{YELLOW};">
<div class="tilenum">${premium_at_risk:,.0f}</div>
<div class="tilecap">ceded premium at risk if they leave</div>
</div>
</div>
""")

left, right = st.columns([1, 1.45], gap="medium")

with left:
    html(f"""
<div class="card">
<div class="eyebrow">Lapse probability</div>
<div class="score" style="color:{band_color};">{proba * 100:.0f}<span class="pc">%</span></div>
<div class="scorecap">chance of cancelling early</div>
<div class="track">
<div class="fill" style="width:{max(proba * 100, 2):.1f}%; background:{band_color};"></div>
</div>
<div class="ticks"><span>0%</span><span>flagged at {threshold:.0%}</span><span>100%</span></div>
<div style="margin-top:1.5rem;">
<span class="pill" style="background:{band_color}22; color:{band_color};">{band}</span>
</div>
<div class="note" style="margin-top:0.9rem;">{action}</div>
</div>
""")

with right:
    html(f"""
<div class="card" style="margin-bottom:0.9rem;">
<div class="eyebrow">{treaty} treaty</div>
<div class="cardhead">How this policy is shared with the reinsurer</div>
<div class="tower">
<div class="layer layer-ceded" style="min-height:{max(26, cession * 0.85):.0f}px;">
<span>Ceded to reinsurer</span><span><b>{cession:.1f}%</b> &nbsp; ${ceded_cover:,.0f}</span>
</div>
<div class="layer layer-retained" style="min-height:{max(26, (100 - cession) * 0.85):.0f}px;">
<span>Retained by the insurer</span><span><b>{100 - cession:.1f}%</b> &nbsp; ${sum_insured - ceded_cover:,.0f}</span>
</div>
</div>
<div class="note" style="margin-top:1.1rem;">
Ceded premium at risk is the annual ceded premium weighted by the lapse probability &mdash;
what this policyholder leaving would cost the reinsurance programme, not just the direct book.
</div>
</div>
""")

    raising = [r.label for r in contrib.head(3).itertuples() if r.shap > 0]
    lowering = [r.label for r in contrib.head(3).itertuples() if r.shap < 0]
    sentence = ""
    if raising:
        sentence += f"Pushing risk up: <b>{', '.join(raising)}</b>. "
    if lowering:
        sentence += f"Holding risk down: <b>{', '.join(lowering)}</b>."

    html(f"""
<div class="card" style="padding-bottom:1.1rem;">
<div class="eyebrow">Model explanation</div>
<div class="cardhead">What drove this score</div>
<div class="readout" style="margin-top:0.7rem;">{sentence}</div>
</div>
""")
    st.pyplot(driver_chart(contrib), use_container_width=True)
    html('<div class="note">Each bar is one factor\'s contribution to this prediction. '
         'Right of the line raises risk, left lowers it.</div>')

with st.expander("All factor contributions"):
    st.dataframe(
        contrib[["label", "shap"]]
        .rename(columns={"label": "Factor", "shap": "Contribution"})
        .round(4),
        use_container_width=True, hide_index=True,
    )
