"""
Streamlit Dashboard — Geo Marketing Lift Measurement
=====================================================
Interactive visualization of synthetic control results for
Mozilla-style geo-level incrementality analysis.

Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import json
import os
import sys

# ── Allow import of core data-generation logic from analysis.py ──
# We re-create data in-app rather than import to keep app self-contained

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

MOZILLA_RED  = "#FF4F5E"
MOZILLA_BLUE = "#00ADEF"
MOZILLA_DARK = "#1C1B1F"
MOZILLA_GRAY = "#6B7280"

N_PERIODS           = 36
INTERVENTION_PERIOD = 24
TREATED_GEOS        = ["DE", "FR", "PL"]
CONTROL_GEOS        = ["NL", "SE", "AT", "BE", "CH", "CZ", "DK", "FI", "HU", "NO", "PT", "RO"]
ALL_GEOS            = TREATED_GEOS + CONTROL_GEOS
TRUE_LIFT           = {"DE": 2.8, "FR": 2.1, "PL": 3.2}
COLORS              = {"DE": MOZILLA_RED, "FR": MOZILLA_BLUE, "PL": "#F59E0B"}

GEO_PARAMS = {
    "DE":    (  9.5, 0.010, 0.65, 0.18),
    "FR":    (  8.2, 0.008, 0.60, 0.15),
    "PL":    (  7.8, 0.012, 0.70, 0.20),
    "NL":    ( 10.1, 0.005, 0.55, 0.14),
    "SE":    (  8.9, 0.007, 0.62, 0.16),
    "AT":    (  9.3, 0.009, 0.58, 0.17),
    "BE":    (  8.6, 0.006, 0.63, 0.15),
    "CH":    ( 10.4, 0.004, 0.52, 0.13),
    "CZ":    (  7.1, 0.011, 0.68, 0.19),
    "DK":    (  9.7, 0.006, 0.57, 0.15),
    "FI":    (  8.4, 0.008, 0.61, 0.16),
    "HU":    (  6.8, 0.013, 0.72, 0.21),
    "NO":    (  9.2, 0.005, 0.56, 0.14),
    "PT":    (  7.5, 0.010, 0.66, 0.18),
    "RO":    (  6.2, 0.014, 0.75, 0.22),
}

@st.cache_data
def generate_panel():
    import scipy.optimize as opt
    dates  = pd.date_range("2022-01-01", periods=N_PERIODS, freq="MS")
    records = []
    for i, geo in enumerate(ALL_GEOS):
        base, trend, phi, noise_sd = GEO_PARAMS[geo]
        local_rng = np.random.default_rng(RNG_SEED + i * 100)
        series = np.zeros(N_PERIODS)
        eps    = local_rng.normal(0, noise_sd, N_PERIODS)
        series[0] = base + eps[0]
        for t in range(1, N_PERIODS):
            series[t] = base + trend*t + phi*(series[t-1]-base-trend*(t-1)) + eps[t]
        counterfactual = series.copy()
        if geo in TREATED_GEOS:
            for t in range(INTERVENTION_PERIOD, N_PERIODS):
                ramp      = min(1.0, (t - INTERVENTION_PERIOD + 1) / 3.0)
                series[t] = counterfactual[t] + TRUE_LIFT[geo] * ramp
        for t, dt in enumerate(dates):
            records.append({
                "date": dt, "geo": geo, "period": t,
                "firefox_share": round(float(series[t]), 4),
                "counterfactual": round(float(counterfactual[t]), 4),
                "is_treated": geo in TREATED_GEOS,
                "is_post": t >= INTERVENTION_PERIOD,
            })
    panel = pd.DataFrame(records)

    spend_rng = np.random.default_rng(RNG_SEED + 999)
    panel["campaign_spend_usd"] = 0.0
    for geo in TREATED_GEOS:
        mask  = (panel["geo"] == geo) & (panel["is_post"])
        n     = mask.sum()
        base_s = {"DE": 180_000, "FR": 150_000, "PL": 120_000}[geo]
        panel.loc[mask, "campaign_spend_usd"] = (
            base_s + spend_rng.normal(0, base_s*0.05, n)
        ).clip(min=0)
    return panel, dates

@st.cache_data
def compute_synthetic_controls(panel_json):
    import scipy.optimize as opt
    panel = pd.read_json(panel_json, orient="records")
    panel["date"] = pd.to_datetime(panel["date"])
    results = {}
    for geo in ALL_GEOS:
        donors = [g for g in ALL_GEOS if g != geo]
        pre    = panel["period"] < INTERVENTION_PERIOD
        t_pre  = panel[(panel["geo"] == geo) & pre].sort_values("period")["firefox_share"].values
        dm     = np.column_stack([
            panel[(panel["geo"] == d) & pre].sort_values("period")["firefox_share"].values
            for d in donors
        ])
        n_d = len(donors)
        res = opt.minimize(
            lambda w: np.sum((t_pre - dm@w)**2), np.ones(n_d)/n_d,
            method="SLSQP",
            bounds=[(0,1)]*n_d,
            constraints={"type":"eq","fun":lambda w:np.sum(w)-1},
            options={"maxiter":2000,"ftol":1e-12}
        )
        full_dm = np.column_stack([
            panel[panel["geo"]==d].sort_values("period")["firefox_share"].values
            for d in donors
        ])
        synth  = full_dm @ res.x
        actual = panel[panel["geo"]==geo].sort_values("period")["firefox_share"].values
        lift   = actual - synth
        results[geo] = {
            "synth":  synth.tolist(),
            "actual": actual.tolist(),
            "lift":   lift.tolist(),
        }
    return results

# ─────────────────────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geo Marketing Lift | Mozilla Demo",
    page_icon="🦊",
    layout="wide",
)

st.title("🦊 Geo-Based Marketing Lift — Synthetic Control Dashboard")
st.caption(
    "**Demo project on fully synthetic data.** "
    "Models the kind of geo-incrementality experiment Mozilla's "
    "Marketing Data Science team would run. No internal data used."
)
st.markdown("---")

# Load data
panel, dates = generate_panel()
panel_json   = panel.to_json(orient="records")
sc_results   = compute_synthetic_controls(panel_json)

# Load pre-computed results if available
if os.path.exists("results.json"):
    with open("results.json") as f:
        precomputed = json.load(f)
else:
    precomputed = None

# ── Sidebar ──
st.sidebar.header("Controls")
selected_geo = st.sidebar.selectbox("Select Geo", ALL_GEOS, index=0)
show_truth   = st.sidebar.checkbox("Show ground-truth counterfactual", value=False)
show_donors  = st.sidebar.checkbox("Show donor pool series", value=True)

# ── Main panel: counterfactual chart ──
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Actual vs Synthetic Control — {selected_geo}")
    res    = sc_results[selected_geo]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    color  = COLORS.get(selected_geo, MOZILLA_GRAY)

    if show_donors:
        for donor in [g for g in ALL_GEOS if g != selected_geo]:
            d_vals = panel[panel["geo"]==donor].sort_values("period")["firefox_share"].values
            ax.plot(dates, d_vals, color=MOZILLA_GRAY, alpha=0.25, lw=0.8)

    ax.plot(dates, res["actual"], color=color, lw=2.5, label=f"{selected_geo} Actual")
    ax.plot(dates, res["synth"],  color=MOZILLA_DARK, lw=2, ls="--", label="Synthetic Control")

    if show_truth and selected_geo in TREATED_GEOS:
        truth = panel[panel["geo"]==selected_geo].sort_values("period")["counterfactual"].values
        ax.plot(dates, truth, color="green", lw=1.5, ls=":", label="True counterfactual")

    ax.axvline(dates[INTERVENTION_PERIOD], color="black", lw=1.5, ls=":", alpha=0.7,
               label="Campaign start (Jan 2024)")

    if selected_geo in TREATED_GEOS:
        ax.fill_between(
            dates[INTERVENTION_PERIOD:],
            np.array(res["synth"])[INTERVENTION_PERIOD:],
            np.array(res["actual"])[INTERVENTION_PERIOD:],
            alpha=0.2, color=color, label="Estimated lift"
        )

    ax.set_xlabel("Date"); ax.set_ylabel("Firefox Share (%)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.35)
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col2:
    st.subheader("Lift Metrics")
    lift_arr   = np.array(res["lift"])
    post_lift  = lift_arr[INTERVENTION_PERIOD:]
    mean_lift  = post_lift.mean()
    cum_lift   = post_lift.sum()
    tag        = "🟢 Treated" if selected_geo in TREATED_GEOS else "⚪ Control"
    st.metric("Geo type", tag)
    st.metric("Mean post-period lift", f"{mean_lift:.3f} pp")
    st.metric("Cumulative lift", f"{cum_lift:.2f} pp-months")

    if precomputed and selected_geo in precomputed.get("geos", {}):
        g = precomputed["geos"][selected_geo]
        st.metric("95% CI", f"[{g['ci_lo']:.3f}, {g['ci_hi']:.3f}]")
        st.metric("Placebo p-value", f"{g['p_value']:.3f}")
        if selected_geo in TREATED_GEOS:
            st.metric("Est. Incremental Installs", f"{g['incremental_installs']:,}")
            st.metric("Cost per Install", f"${g['cpi']:.2f}")

st.markdown("---")

# ── Summary table ──
st.subheader("Cross-Geo Lift Summary")
rows = []
for geo in TREATED_GEOS:
    r   = sc_results[geo]
    la  = np.array(r["lift"])[INTERVENTION_PERIOD:]
    row = {"Geo": geo, "Mean Lift (pp)": round(la.mean(), 3),
           "Cumulative (pp-months)": round(la.sum(), 2),
           "True Lift (pp)": TRUE_LIFT[geo]}
    if precomputed and geo in precomputed.get("geos", {}):
        g = precomputed["geos"][geo]
        row["95% CI Lo"] = g["ci_lo"]
        row["95% CI Hi"] = g["ci_hi"]
        row["p-value"]   = g["p_value"]
        row["CPI ($)"]   = g["cpi"]
    rows.append(row)
st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.markdown("---")

# ── Cumulative lift chart ──
st.subheader("Cumulative Lift Curves — All Treated Geos")
fig2, ax2 = plt.subplots(figsize=(12, 4))
for geo in TREATED_GEOS:
    r   = sc_results[geo]
    cum = np.cumsum(np.array(r["lift"])[INTERVENTION_PERIOD:])
    ax2.plot(dates[INTERVENTION_PERIOD:], cum, color=COLORS[geo], lw=2.5, label=geo)
ax2.axhline(0, color=MOZILLA_DARK, lw=0.8, ls=":")
ax2.set_ylabel("Cumulative Lift (pp-months)")
ax2.set_title("Cumulative Incremental Lift Post-Campaign Launch")
ax2.legend(); ax2.grid(True, alpha=0.35)
ax2.set_facecolor("white"); fig2.patch.set_facecolor("white")
st.pyplot(fig2, use_container_width=True)
plt.close()

# ── SHAP chart ──
st.markdown("---")
st.subheader("SHAP Feature Importance — XGBoost Companion Model")
if os.path.exists("charts/05_shap_importance.png"):
    st.image("charts/05_shap_importance.png", use_container_width=True)
    st.caption(
        "SHAP values from XGBoost model predicting post-campaign Firefox share. "
        "Pre-campaign baseline and campaign spend are the dominant predictors."
    )

st.markdown("---")
st.caption(
    "All data synthetic. Geo parameters modeled after publicly-known browser "
    "market patterns in EU markets. Campaign spend is simulated. "
    "Source: Portfolio project for Mozilla Senior Marketing Data Scientist role."
)
