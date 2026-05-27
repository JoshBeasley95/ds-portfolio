"""
app.py  —  Guest Insurance Intent Scorer
Streamlit micro-app: input a hypothetical guest profile,
get a calibrated uptake probability + top-3 SHAP drivers.

Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import shap
import joblib
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Feature list (must match analysis.py order) ──────────────
FEATURES = [
    "Age", "EmploymentType", "GraduateOrNot", "AnnualIncome",
    "FamilyMembers", "ChronicDiseases", "FrequentFlyer", "EverTravelledAbroad",
    "IncomeBucket", "TripRiskScore", "Income_FF", "AgeGroup", "HighDependents",
]

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("insurance_intent_model.pkl")

model = load_model()

# ── Feature engineering helper ────────────────────────────────
def engineer(age, emp_type, grad, income, family, chronic, ff, abroad):
    emp_enc   = 1 if emp_type == "Government Sector" else 0
    grad_enc  = 1 if grad == "Yes" else 0
    ff_enc    = 1 if ff == "Yes" else 0
    abroad_enc= 1 if abroad == "Yes" else 0

    income_bucket = int(pd.cut([income],
        bins=[0, 400_000, 700_000, 1_000_000, 1_400_000, float("inf")],
        labels=[0, 1, 2, 3, 4])[0])

    trip_risk = (chronic * 0.35 + abroad_enc * 0.25 +
                 ff_enc  * 0.20 + (family / 9) * 0.20)
    income_ff = income * ff_enc / 1e6
    age_group = int(pd.cut([age], bins=[0, 25, 35, 45, 100],
                            labels=[0, 1, 2, 3])[0])
    high_dep  = int(family >= 6)

    row = [age, emp_enc, grad_enc, income, family, chronic, ff_enc, abroad_enc,
           income_bucket, trip_risk, income_ff, age_group, high_dep]
    return np.array(row, dtype=float).reshape(1, -1)

# ── SHAP explainer ────────────────────────────────────────────
@st.cache_resource
def get_explainer(_model):
    # Extract the underlying XGB estimators from CalibratedClassifierCV
    base = _model.calibrated_classifiers_[0].estimator
    return shap.TreeExplainer(base)

explainer = get_explainer(model)

# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Guest Insurance Intent Scorer",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Guest Insurance Intent Scorer")
st.caption("Airbnb GTI Algorithms · Portfolio Demo · No proprietary data used")

st.markdown("""
Enter a hypothetical guest's profile to receive a calibrated **insurance purchase
probability** and the **top SHAP drivers** behind that score.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    age         = st.slider("Age", 18, 65, 32)
    income      = st.number_input("Annual Income (INR)", 300_000, 1_800_000,
                                   700_000, step=50_000)
    family      = st.slider("Family Members", 2, 9, 4)
    chronic     = st.selectbox("Chronic Disease", [0, 1],
                                format_func=lambda x: "Yes" if x else "No")

with col2:
    emp_type    = st.selectbox("Employment Type",
                               ["Private Sector/Self Employed", "Government Sector"])
    grad        = st.selectbox("College Graduate", ["Yes", "No"])
    ff          = st.selectbox("Frequent Flyer", ["No", "Yes"])
    abroad      = st.selectbox("Ever Travelled Abroad", ["No", "Yes"])

st.divider()

if st.button("📊 Score This Guest", type="primary"):
    X_guest = engineer(age, emp_type, grad, income, family, chronic, ff, abroad)

    prob = model.predict_proba(X_guest)[0, 1]

    # SHAP for top drivers
    shap_vals = explainer.shap_values(X_guest)[0]
    shap_df   = pd.DataFrame({
        "Feature": FEATURES,
        "SHAP":    shap_vals,
        "Value":   X_guest[0],
    }).sort_values("SHAP", key=abs, ascending=False)

    top3 = shap_df.head(3)

    # ── Result display ──────────────────────────────────────
    st.subheader("Scoring Result")

    col_prob, col_tier = st.columns(2)
    with col_prob:
        st.metric("Insurance Intent Probability", f"{prob:.1%}")
    with col_tier:
        tier = ("🔴 High Intent"   if prob >= 0.45
           else "🟡 Medium Intent" if prob >= 0.25
           else "🟢 Low Intent")
        st.metric("Intent Tier", tier)

    if prob >= 0.45:
        st.success("Recommendation: surface the insurance offer prominently at checkout.")
    elif prob >= 0.25:
        st.info("Recommendation: surface a soft nudge or contextual banner.")
    else:
        st.warning("Recommendation: suppress insurance offer to avoid annoyance.")

    st.subheader("Top 3 SHAP Drivers")
    for _, row in top3.iterrows():
        direction = "↑ increases" if row["SHAP"] > 0 else "↓ decreases"
        st.markdown(
            f"**{row['Feature']}** = `{row['Value']:.3g}` &nbsp; "
            f"| SHAP = `{row['SHAP']:+.3f}` → {direction} intent"
        )

    # Waterfall chart
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [("#FF5A5F" if s > 0 else "#00A699") for s in shap_df.head(8)["SHAP"]]
    ax.barh(shap_df.head(8)["Feature"], shap_df.head(8)["SHAP"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP value")
    ax.set_title("Per-Guest SHAP Contributions", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()
st.caption(
    "⚠️ Built on synthetic data for portfolio demonstration purposes. "
    "Not trained on Airbnb guest data."
)
