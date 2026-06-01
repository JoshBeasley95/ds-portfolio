"""
app.py — Streamlit: Lifecycle Segment Lift Explorer
Cohort selector shows predicted conversion lift + top SHAP drivers.
Dark theme via .streamlit/config.toml
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb

from generate_data import generate_dataset

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lifecycle Segment Lift",
    page_icon="📊",
    layout="wide",
)

DARK_BG = "#0E1117"
ACCENT  = "#E87C3E"
ACCENT2 = "#4C9ED9"

def dark_mpl():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor":   DARK_BG,
        "axes.edgecolor":   "#2A2D3A",
        "axes.labelcolor":  "#FAFAFA",
        "xtick.color":      "#FAFAFA",
        "ytick.color":      "#FAFAFA",
        "text.color":       "#FAFAFA",
        "grid.color":       "#2A2D3A",
        "grid.linewidth":   0.5,
        "axes.grid":        True,
    })

dark_mpl()

# ── Data + Model (cached) ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = generate_dataset()
    df["plan_pro"] = (df["plan_type"] == "pro").astype(int)
    df["ind_tech"]  = (df["industry"] == "tech").astype(int)
    df["ind_fin"]   = (df["industry"] == "finance").astype(int)
    df["ind_hlth"]  = (df["industry"] == "healthcare").astype(int)
    return df

@st.cache_resource
def train_model(df):
    ps_features = ["sessions_last_14d", "messages_sent", "days_since_signup",
                   "plan_pro", "ind_tech", "ind_fin", "ind_hlth"]
    X_ps = df[ps_features].values
    T = df["received_onboarding_email"].values.astype(int)

    scaler = StandardScaler()
    X_ps_s = scaler.fit_transform(X_ps)
    ps_model = LogisticRegression(max_iter=500, random_state=42)
    ps_model.fit(X_ps_s, T)
    df = df.copy()
    df["propensity_score"] = ps_model.predict_proba(X_ps_s)[:, 1]

    treated = df[df["received_onboarding_email"] == 1]
    control = df[df["received_onboarding_email"] == 0]
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[["propensity_score"]].values)
    _, indices = nn.kneighbors(treated[["propensity_score"]].values)
    matched = pd.concat([treated, control.iloc[indices.flatten()]], ignore_index=True)

    model_features = ps_features + ["received_onboarding_email"]
    X_m = matched[model_features].values
    y_m = matched["converted_to_paid"].values.astype(int)
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        use_label_encoder=False, verbosity=0, eval_metric="logloss"
    )
    clf.fit(X_m, y_m)
    explainer = shap.TreeExplainer(clf)
    return clf, explainer, model_features, df

df = load_data()
clf, explainer, model_features, df_with_ps = train_model(df)

FEAT_LABELS = {
    "sessions_last_14d": "Sessions (14d)",
    "messages_sent": "Messages Sent",
    "days_since_signup": "Days Since Signup",
    "plan_pro": "Plan: Pro",
    "ind_tech": "Industry: Tech",
    "ind_fin": "Industry: Finance",
    "ind_hlth": "Industry: Healthcare",
    "received_onboarding_email": "Email Received",
}

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📊 Lifecycle Segment Lift Explorer")
st.markdown(
    "Explore how the **onboarding email** drives conversion across user segments. "
    "Uses propensity-score matching to estimate causal lift, and XGBoost + SHAP "
    "to explain which behavioral signals matter most."
)
st.markdown("---")

# ══ Layout: 3 columns ════════════════════════════════════════════════════════
col_left, col_mid, col_right = st.columns([1, 1.8, 1.5])

# ── LEFT: Cohort Filters ─────────────────────────────────────────────────────
with col_left:
    st.subheader("🔎 Filter Cohort")

    plan_sel = st.selectbox("Plan Type", ["All", "free", "pro"])
    industry_sel = st.selectbox("Industry", ["All", "tech", "finance", "healthcare", "other"])
    seg_labels = ["All"] + sorted(df["segment_label"].dropna().unique().tolist()) if "segment_label" in df.columns else ["All"]

    sessions_range = st.slider(
        "Sessions in last 14 days", int(df["sessions_last_14d"].min()),
        int(df["sessions_last_14d"].max()),
        (0, int(df["sessions_last_14d"].max()))
    )
    messages_range = st.slider(
        "Messages Sent", int(df["messages_sent"].min()),
        int(df["messages_sent"].max()),
        (0, int(df["messages_sent"].max()))
    )

    # Apply filters
    mask = (
        (df["sessions_last_14d"] >= sessions_range[0]) &
        (df["sessions_last_14d"] <= sessions_range[1]) &
        (df["messages_sent"] >= messages_range[0]) &
        (df["messages_sent"] <= messages_range[1])
    )
    if plan_sel != "All":
        mask &= df["plan_type"] == plan_sel
    if industry_sel != "All":
        mask &= df["industry"] == industry_sel

    cohort = df_with_ps[mask].copy()
    st.metric("Cohort Size", f"{len(cohort):,}")
    st.metric("Baseline Conversion", f"{cohort['converted_to_paid'].mean():.1%}")

# ── MID: Lift Estimates ───────────────────────────────────────────────────────
with col_mid:
    st.subheader("⚡ Predicted Conversion Lift")

    if len(cohort) < 10:
        st.warning("Too few users in cohort. Adjust filters.")
    else:
        X_t1 = cohort[model_features].copy()
        X_t0 = cohort[model_features].copy()
        X_t1["received_onboarding_email"] = 1
        X_t0["received_onboarding_email"] = 0

        p1 = clf.predict_proba(X_t1.values)[:, 1]
        p0 = clf.predict_proba(X_t0.values)[:, 1]
        lift = p1 - p0
        cohort["predicted_lift"] = lift

        avg_lift = lift.mean()
        p0_mean  = p0.mean()
        p1_mean  = p1.mean()

        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Lift (ΔP)", f"{avg_lift:+.3f}")
        m2.metric("P(convert | no email)", f"{p0_mean:.3f}")
        m3.metric("P(convert | email)", f"{p1_mean:.3f}")

        # Lift distribution
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor(DARK_BG)
        ax.hist(lift, bins=40, color=ACCENT, edgecolor="#0E1117", alpha=0.85)
        ax.axvline(avg_lift, color=ACCENT2, linestyle="--", linewidth=2,
                   label=f"Mean={avg_lift:+.3f}")
        ax.set_title("Distribution of Predicted Email Lift")
        ax.set_xlabel("ΔP(Conversion)")
        ax.set_ylabel("Users")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Lift by plan (within cohort)
        if cohort["plan_type"].nunique() > 1:
            lift_plan = cohort.groupby("plan_type")["predicted_lift"].mean()
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            fig2.patch.set_facecolor(DARK_BG)
            colors = [ACCENT if v == lift_plan.max() else ACCENT2 for v in lift_plan.values]
            ax2.bar(lift_plan.index, lift_plan.values, color=colors, edgecolor="#2A2D3A")
            ax2.set_title("Avg Lift by Plan (cohort)")
            ax2.set_ylabel("Avg Lift")
            for i, v in enumerate(lift_plan.values):
                ax2.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=10)
            st.pyplot(fig2, use_container_width=True)
            plt.close()

# ── RIGHT: SHAP Drivers ───────────────────────────────────────────────────────
with col_right:
    st.subheader("🧠 Top Conversion Drivers (SHAP)")

    if len(cohort) >= 10:
        sample_size = min(500, len(cohort))
        cohort_sample = cohort.sample(sample_size, random_state=42) if len(cohort) > sample_size else cohort
        X_shap = cohort_sample[model_features].values
        shap_vals = explainer.shap_values(X_shap)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        feat_imp = pd.Series(mean_abs, index=[FEAT_LABELS[f] for f in model_features])
        feat_imp = feat_imp.sort_values()

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        fig3.patch.set_facecolor(DARK_BG)
        colors = [ACCENT if i >= len(feat_imp) - 3 else ACCENT2 for i in range(len(feat_imp))]
        ax3.barh(feat_imp.index, feat_imp.values, color=colors, edgecolor="#2A2D3A")
        ax3.set_title(f"Mean |SHAP| — Cohort (n={len(cohort_sample):,})")
        ax3.set_xlabel("Mean |SHAP Value|")
        st.pyplot(fig3, use_container_width=True)
        plt.close()

        # Top feature detail
        top_feat = feat_imp.idxmax()
        st.info(f"**Top driver:** {top_feat}  \n"
                f"Mean |SHAP| = {feat_imp.max():.3f}")

        # Conversion rate breakdown
        st.markdown("**Cohort Breakdown**")
        breakdown = cohort.groupby("plan_type").agg(
            n=("user_id", "count"),
            conv_rate=("converted_to_paid", "mean"),
            avg_lift=("predicted_lift", "mean"),
        ).round(3)
        st.dataframe(breakdown, use_container_width=True)

# ── Bottom: Full Segment Table ────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Full Segment Lift Comparison")

all_X_t1 = df_with_ps[model_features].copy()
all_X_t0 = df_with_ps[model_features].copy()
all_X_t1["received_onboarding_email"] = 1
all_X_t0["received_onboarding_email"] = 0
df_with_ps = df_with_ps.copy()
df_with_ps["predicted_lift"] = (
    clf.predict_proba(all_X_t1.values)[:, 1] -
    clf.predict_proba(all_X_t0.values)[:, 1]
)

seg_table = df_with_ps.groupby("segment_label").agg(
    n=("user_id", "count"),
    conversion_rate=("converted_to_paid", "mean"),
    email_rate=("received_onboarding_email", "mean"),
    avg_predicted_lift=("predicted_lift", "mean"),
).round(3).sort_values("avg_predicted_lift", ascending=False)

st.dataframe(seg_table, use_container_width=True)

st.caption(
    "Demonstration on synthetic data only. Methodology mirrors real lifecycle "
    "experiment design: propensity-score matching removes selection bias before "
    "estimating causal lift."
)
