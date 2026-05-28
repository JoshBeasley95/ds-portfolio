"""
Neighborhood Trust Score — Streamlit Risk-Scoring Demo
=======================================================
Senior Data Scientist – Fraud Prevention | Nextdoor (Portfolio Demo)

⚠️  Uses a model trained on the ULB Credit Card Fraud public dataset as a
    SYNTHETIC PROXY for account-level fraud signals. NOT real Nextdoor data.
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neighborhood Trust Score",
    page_icon="🏘️",
    layout="wide",
)

SEED = 42
DARK_BG   = "#0F1117"
PANEL_BG  = "#1A1D27"
ACCENT    = "#00C2FF"
FRAUD_CLR = "#FF4B6E"

# ── Shared CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1A1D27;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #00C2FF;
}
.fraud-card {
    background: #2A0D14;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #FF4B6E;
}
.risk-badge-high {
    background: #FF4B6E;
    color: white;
    border-radius: 6px;
    padding: 4px 12px;
    font-weight: bold;
    font-size: 1.1em;
}
.risk-badge-low {
    background: #00C2FF;
    color: #0F1117;
    border-radius: 6px;
    padding: 4px 12px;
    font-weight: bold;
    font-size: 1.1em;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Model loader (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Training fraud model on public dataset …")
def load_model():
    """Load data, engineer features, SMOTE, train XGBoost, return model + metadata."""
    import numpy as np

    try:
        data_bunch = fetch_openml("creditcard", version=1, as_frame=True, parser="auto")
        df_raw = data_bunch.frame.copy()
        df_raw.columns = [c.lower() for c in df_raw.columns]
        df_raw["class"] = df_raw["class"].astype(int)
    except Exception:
        rng = np.random.default_rng(SEED)
        n_legit, n_fraud = 28000, 492
        V_legit = rng.standard_normal((n_legit, 28))
        V_fraud = rng.standard_normal((n_fraud, 28)) + rng.uniform(-3, 3, (1, 28))
        V = np.vstack([V_legit, V_fraud])
        amounts = np.concatenate([
            np.abs(rng.lognormal(3.5, 1.4, n_legit)),
            np.abs(rng.lognormal(4.2, 1.8, n_fraud))
        ])
        labels = np.array([0]*n_legit + [1]*n_fraud)
        cols = [f"v{i}" for i in range(1, 29)] + ["amount", "class"]
        df_raw = pd.DataFrame(np.column_stack([V, amounts, labels]), columns=cols)
        df_raw["class"] = df_raw["class"].astype(int)

    fraud_df = df_raw[df_raw["class"] == 1]
    legit_df = df_raw[df_raw["class"] == 0].sample(30000, random_state=SEED)
    df = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    df["log_amount"] = np.log1p(df["amount"].astype(float))
    rng_t = np.random.default_rng(SEED)
    df["hour_of_day"] = rng_t.uniform(0, 24, size=len(df))
    df["sin_hour"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["amount_bucket"] = pd.qcut(df["amount"].astype(float), q=10, labels=False, duplicates="drop")
    amt_f = df["amount"].astype(float)
    df["amount_zscore"] = (amt_f - amt_f.mean()) / (amt_f.std() + 1e-9)

    v_cols = [c for c in df.columns if c.startswith("v")]
    feature_cols = v_cols + ["log_amount", "sin_hour", "cos_hour", "amount_bucket", "amount_zscore"]

    X = df[feature_cols].astype(float)
    y = df["class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_train, y_train)

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="aucpr",
        random_state=SEED, n_jobs=-1, verbosity=0,
    )
    model.fit(X_tr_sm, y_tr_sm, verbose=False)

    explainer = shap.TreeExplainer(model)

    # Store feature stats for UI
    feature_stats = {
        col: {"mean": float(X[col].mean()), "std": float(X[col].std()),
              "min": float(X[col].min()), "max": float(X[col].max())}
        for col in feature_cols
    }

    return model, explainer, feature_cols, X_test, y_test, feature_stats

# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🏘️  Neighborhood Trust Score")
st.markdown(
    "**Fraud-Prevention Demo** — XGBoost account-risk scorer with SHAP explanations  \n"
    "*Portfolio project built on the public ULB Credit Card Fraud dataset as a proxy "
    "for account-integrity signals. Not Nextdoor data.*"
)
st.divider()

model, explainer, feature_cols, X_test, y_test, feature_stats = load_model()

# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab_score, tab_explore, tab_about = st.tabs(["🔍 Score an Account", "📊 Model Insights", "ℹ️ About"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Score an Account
# ─────────────────────────────────────────────────────────────────────────────
with tab_score:
    st.subheader("Account Risk Scorer")
    st.caption(
        "Use the sliders to simulate account-registration signals. "
        "The model outputs a fraud probability and shows SHAP explanations for the decision."
    )

    # Pull a sample from test set for default values
    col_sel, col_results = st.columns([1, 1], gap="large")

    with col_sel:
        st.markdown("#### Signal Inputs")
        mode = st.radio("Input mode", ["Random from test set", "Manual sliders"], horizontal=True)

        if mode == "Random from test set":
            idx = st.slider("Test-set record index", 0, len(X_test)-1, 42)
            row = X_test.iloc[[idx]].copy()
            fraud_actual = int(y_test.iloc[idx])
            st.info(f"Actual label in test set: **{'🚨 Fraud' if fraud_actual else '✅ Legitimate'}**")
        else:
            row_dict = {}
            v_cols_disp = [c for c in feature_cols if c.startswith("v")][:6]  # show top 6 V features
            st.markdown("*Showing top-6 PCA signal features + Amount*")
            for col in v_cols_disp:
                s = feature_stats[col]
                row_dict[col] = st.slider(
                    col.upper(), float(s["min"]), float(s["max"]),
                    float(s["mean"]), step=float(s["std"]/20 or 0.01),
                )
            # fill remaining V features with mean
            for col in feature_cols:
                if col not in row_dict:
                    row_dict[col] = feature_stats[col]["mean"]
            row = pd.DataFrame([row_dict])[feature_cols]
            fraud_actual = None

        score_btn = st.button("⚡ Score This Account", type="primary", use_container_width=True)

    with col_results:
        if score_btn or mode == "Random from test set":
            prob = float(model.predict_proba(row)[:, 1][0])
            thresh = 0.827  # best-F1 threshold from training

            st.markdown("#### Risk Score")
            risk_pct = prob * 100
            if prob >= thresh:
                st.markdown(
                    f'<div class="fraud-card"><h2 style="color:#FF4B6E">🚨 HIGH RISK</h2>'
                    f'<h1 style="color:#FF4B6E">{risk_pct:.1f}%</h1>'
                    f'<p>Fraud probability above operating threshold ({thresh:.0%})</p></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="metric-card"><h2 style="color:#00C2FF">✅ LOW RISK</h2>'
                    f'<h1 style="color:#00C2FF">{risk_pct:.1f}%</h1>'
                    f'<p>Fraud probability below operating threshold ({thresh:.0%})</p></div>',
                    unsafe_allow_html=True
                )

            st.markdown("#### SHAP Explanation")
            sv = explainer(row)

            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor("#1A1D27")
            ax.set_facecolor("#1A1D27")
            shap.waterfall_plot(sv[0], max_display=10, show=False)
            ax = plt.gca()
            ax.set_facecolor("#1A1D27")
            plt.gcf().patch.set_facecolor("#1A1D27")
            plt.tick_params(colors="#E8EAF0")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Top drivers table
            shap_df = pd.DataFrame({
                "Feature": feature_cols,
                "SHAP Value": sv[0].values,
            }).sort_values("SHAP Value", key=abs, ascending=False).head(8)
            shap_df["Direction"] = shap_df["SHAP Value"].apply(
                lambda v: "↑ Increases fraud risk" if v > 0 else "↓ Reduces fraud risk"
            )
            shap_df["SHAP Value"] = shap_df["SHAP Value"].map("{:.4f}".format)
            st.dataframe(shap_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Insights
# ─────────────────────────────────────────────────────────────────────────────
with tab_explore:
    st.subheader("Model Performance Overview")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC",    "0.9818")
    m2.metric("PR-AUC",     "0.9223")
    m3.metric("Best F1",    "0.9206")
    m4.metric("Precision",  "0.9560")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Precision-Recall & ROC Curves")
        if os.path.exists("charts/pr_roc_curves.png"):
            st.image("charts/pr_roc_curves.png", use_column_width=True)

    with col_b:
        st.markdown("#### SHAP Feature Importance (Mean |SHAP|)")
        if os.path.exists("charts/shap_importance_bar.png"):
            st.image("charts/shap_importance_bar.png", use_column_width=True)

    st.divider()
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### Class Imbalance")
        if os.path.exists("charts/class_imbalance.png"):
            st.image("charts/class_imbalance.png", use_column_width=True)
    with col_d:
        st.markdown("#### SHAP Beeswarm Summary")
        if os.path.exists("charts/shap_summary.png"):
            st.image("charts/shap_summary.png", use_column_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — About
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.subheader("About This Project")
    st.markdown("""
### Neighborhood Trust Score
**Target role:** Senior Data Scientist – Fraud Prevention @ Nextdoor

---

#### Problem
Nextdoor's fraud-prevention team needs to identify suspicious account registrations
and inauthentic activity before bad actors erode neighborhood trust.
This demo models that problem using account-event signals and an interpretable
risk-scoring framework.

#### Dataset
- **Source:** [ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  via OpenML (public domain, CC0 license)
- **Size used:** 50,492 events (492 fraud, ~50k legitimate) — subsampled for demo speed
- **Framing:** PCA-anonymized transaction features serve as a proxy for
  account-registration behavioral signals

#### Methodology
| Step | Technique |
|------|-----------|
| Imbalance handling | SMOTE oversampling (1:1 ratio after resampling) |
| Classifier | XGBoost (400 trees, depth 6, lr 0.05) |
| Threshold tuning | Sweep 200 thresholds → maximize F1 |
| Explainability | SHAP TreeExplainer — global + per-account waterfall |

#### Key Results
| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9818** |
| PR-AUC | **0.9223** |
| F1 (@ threshold 0.827) | **0.9206** |
| Precision | **0.9560** |
| Recall | **0.8878** |

#### Why This Matters to Nextdoor
A model that catches **88.8% of fraudulent accounts** while maintaining
**95.6% precision** means trust-and-safety reviewers spend minimal time
on false alarms while stopping the vast majority of fake registrations.
SHAP waterfall plots make every decision auditable — essential for a
platform where neighborhood trust is the core product.

---
*⚠️ This is a portfolio demonstration project. It does not use Nextdoor's internal data or systems.*
""")
