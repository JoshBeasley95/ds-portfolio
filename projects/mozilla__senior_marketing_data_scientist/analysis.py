"""
Geo-Based Marketing Lift Measurement with Synthetic Control
===========================================================
Demonstrates CausalImpact-style incrementality measurement for a
hypothetical Firefox geo-experiment using fully synthetic panel data.

ALL DATA IS SYNTHETIC — generated in-code to model the structure of
real geo-experiments Mozilla's Marketing Data Science team would run.
No internal or proprietary Mozilla data is used.

Author: Portfolio Project
Date: 2026-05-27
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import shap
import json
import os

# ─────────────────────────────────────────────────────────────
# 0. REPRODUCIBILITY & STYLE
# ─────────────────────────────────────────────────────────────
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

MOZILLA_RED   = "#FF4F5E"
MOZILLA_BLUE  = "#00ADEF"
MOZILLA_DARK  = "#1C1B1F"
MOZILLA_GRAY  = "#6B7280"
MOZILLA_LIGHT = "#F3F4F6"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": MOZILLA_DARK,
    "axes.labelcolor": MOZILLA_DARK,
    "text.color": MOZILLA_DARK,
    "xtick.color": MOZILLA_DARK,
    "ytick.color": MOZILLA_DARK,
    "grid.color": "#E5E7EB",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
    "font.size": 11,
})

os.makedirs("charts", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Generating synthetic geo-panel data")
print("=" * 60)

# Time span: 36 monthly periods (Jan 2022 – Dec 2024)
N_PERIODS   = 36
N_CONTROL   = 12
N_TREATED   = 3
INTERVENTION_PERIOD = 24   # Month index where campaign starts (Jan 2024)
CAMPAIGN_DURATION   = 12   # 12-month post period

dates = pd.date_range("2022-01-01", periods=N_PERIODS, freq="MS")

# Geo definitions
TREATED_GEOS = ["DE", "FR", "PL"]
CONTROL_GEOS = ["NL", "SE", "AT", "BE", "CH", "CZ", "DK", "FI", "HU", "NO", "PT", "RO"]
ALL_GEOS     = TREATED_GEOS + CONTROL_GEOS

# Geo-level baseline Firefox share parameters (realistic 2022 levels)
GEO_PARAMS = {
    #          baseline  trend     AR1_phi  noise_sd
    "DE":    (  9.5,    0.010,     0.65,    0.18),
    "FR":    (  8.2,    0.008,     0.60,    0.15),
    "PL":    (  7.8,    0.012,     0.70,    0.20),
    "NL":    ( 10.1,    0.005,     0.55,    0.14),
    "SE":    (  8.9,    0.007,     0.62,    0.16),
    "AT":    (  9.3,    0.009,     0.58,    0.17),
    "BE":    (  8.6,    0.006,     0.63,    0.15),
    "CH":    ( 10.4,    0.004,     0.52,    0.13),
    "CZ":    (  7.1,    0.011,     0.68,    0.19),
    "DK":    (  9.7,    0.006,     0.57,    0.15),
    "FI":    (  8.4,    0.008,     0.61,    0.16),
    "HU":    (  6.8,    0.013,     0.72,    0.21),
    "NO":    (  9.2,    0.005,     0.56,    0.14),
    "PT":    (  7.5,    0.010,     0.66,    0.18),
    "RO":    (  6.2,    0.014,     0.75,    0.22),
}

# True lift injected into treated geos (pp above counterfactual)
TRUE_LIFT = {"DE": 2.8, "FR": 2.1, "PL": 3.2}

def simulate_geo_series(geo, seed_offset=0):
    """Simulate AR(1) + trend Firefox share series."""
    base, trend, phi, noise_sd = GEO_PARAMS[geo]
    local_rng = np.random.default_rng(RNG_SEED + seed_offset)
    series = np.zeros(N_PERIODS)
    eps = local_rng.normal(0, noise_sd, N_PERIODS)
    series[0] = base + eps[0]
    for t in range(1, N_PERIODS):
        series[t] = base + trend * t + phi * (series[t-1] - base - trend*(t-1)) + eps[t]
    return series

# Build panel
records = []
for i, geo in enumerate(ALL_GEOS):
    raw = simulate_geo_series(geo, seed_offset=i*100)
    counterfactual = raw.copy()
    
    # Inject lift for treated geos in post-period
    if geo in TREATED_GEOS:
        lift_val = TRUE_LIFT[geo]
        # Ramp up over first 3 months, sustained thereafter
        for t in range(INTERVENTION_PERIOD, N_PERIODS):
            ramp = min(1.0, (t - INTERVENTION_PERIOD + 1) / 3.0)
            raw[t] = counterfactual[t] + lift_val * ramp
    
    for t, dt in enumerate(dates):
        records.append({
            "date": dt,
            "geo": geo,
            "period": t,
            "firefox_share": round(float(raw[t]), 4),
            "counterfactual_share": round(float(counterfactual[t]), 4),
            "is_treated": geo in TREATED_GEOS,
            "is_post": t >= INTERVENTION_PERIOD,
        })

panel = pd.DataFrame(records)

# Add synthetic spend overlay for treated geos
spend_rng = np.random.default_rng(RNG_SEED + 999)
panel["campaign_spend_usd"] = 0.0
for geo in TREATED_GEOS:
    mask = (panel["geo"] == geo) & (panel["is_post"])
    n = mask.sum()
    base_spend = {"DE": 180_000, "FR": 150_000, "PL": 120_000}[geo]
    panel.loc[mask, "campaign_spend_usd"] = (
        base_spend + spend_rng.normal(0, base_spend * 0.05, n)
    ).clip(min=0)

# Add geo-level features (population proxy, tech-savviness index)
GEO_FEATURES = {
    "DE": {"pop_mm": 83.2, "broadband_pct": 91.0, "tech_index": 0.82},
    "FR": {"pop_mm": 68.0, "broadband_pct": 88.5, "tech_index": 0.78},
    "PL": {"pop_mm": 37.9, "broadband_pct": 82.0, "tech_index": 0.71},
    "NL": {"pop_mm": 17.9, "broadband_pct": 95.0, "tech_index": 0.88},
    "SE": {"pop_mm": 10.5, "broadband_pct": 93.0, "tech_index": 0.87},
    "AT": {"pop_mm":  9.1, "broadband_pct": 90.0, "tech_index": 0.83},
    "BE": {"pop_mm": 11.6, "broadband_pct": 89.0, "tech_index": 0.80},
    "CH": {"pop_mm":  8.7, "broadband_pct": 94.0, "tech_index": 0.89},
    "CZ": {"pop_mm": 10.9, "broadband_pct": 85.0, "tech_index": 0.75},
    "DK": {"pop_mm":  5.9, "broadband_pct": 94.0, "tech_index": 0.90},
    "FI": {"pop_mm":  5.5, "broadband_pct": 93.0, "tech_index": 0.88},
    "HU": {"pop_mm":  9.7, "broadband_pct": 78.0, "tech_index": 0.68},
    "NO": {"pop_mm":  5.4, "broadband_pct": 95.0, "tech_index": 0.91},
    "PT": {"pop_mm": 10.3, "broadband_pct": 80.0, "tech_index": 0.72},
    "RO": {"pop_mm": 19.0, "broadband_pct": 72.0, "tech_index": 0.62},
}
for feat, vals in pd.DataFrame(GEO_FEATURES).T.reset_index().rename(columns={"index": "geo"}).items():
    if feat != "geo":
        pass
feat_df = pd.DataFrame(GEO_FEATURES).T.reset_index().rename(columns={"index": "geo"})
panel = panel.merge(feat_df, on="geo", how="left")

print(f"  Panel shape: {panel.shape}")
print(f"  Treated geos: {TREATED_GEOS}")
print(f"  Control geos: {CONTROL_GEOS}")
print(f"  Intervention period: {dates[INTERVENTION_PERIOD].strftime('%b %Y')}")
print(f"  True lift injected: DE={TRUE_LIFT['DE']}pp, FR={TRUE_LIFT['FR']}pp, PL={TRUE_LIFT['PL']}pp")

# ─────────────────────────────────────────────────────────────
# 2. EDA — TIME-SERIES TRENDS & PARALLEL TRENDS CHECK
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: EDA — time-series trends & parallel trends check")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# Left: All geos raw share
ax = axes[0]
for geo in CONTROL_GEOS:
    sub = panel[panel["geo"] == geo]
    ax.plot(sub["date"], sub["firefox_share"], color=MOZILLA_GRAY, alpha=0.5, lw=1.2)
for geo in TREATED_GEOS:
    sub = panel[panel["geo"] == geo]
    color = {"DE": MOZILLA_RED, "FR": MOZILLA_BLUE, "PL": "#F59E0B"}[geo]
    ax.plot(sub["date"], sub["firefox_share"], color=color, lw=2.2, label=f"{geo} (treated)")
ax.axvline(dates[INTERVENTION_PERIOD], color=MOZILLA_DARK, ls="--", lw=1.5, label="Campaign start")
ax.set_title("Firefox Market Share by Geo\n(Treated vs Control)", fontsize=12, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Firefox Share (%)")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.4)

# Right: Indexed to pre-period mean (parallel trends viz)
ax = axes[1]
pre_panel = panel[panel["period"] < INTERVENTION_PERIOD]
pre_means = pre_panel.groupby("geo")["firefox_share"].mean()
for geo in CONTROL_GEOS:
    sub = panel[panel["geo"] == geo].copy()
    sub["indexed"] = sub["firefox_share"] / pre_means[geo] * 100
    ax.plot(sub["date"], sub["indexed"], color=MOZILLA_GRAY, alpha=0.4, lw=1.0)
for geo in TREATED_GEOS:
    sub = panel[panel["geo"] == geo].copy()
    sub["indexed"] = sub["firefox_share"] / pre_means[geo] * 100
    color = {"DE": MOZILLA_RED, "FR": MOZILLA_BLUE, "PL": "#F59E0B"}[geo]
    ax.plot(sub["date"], sub["indexed"], color=color, lw=2.2, label=geo)
ax.axhline(100, color=MOZILLA_DARK, ls=":", lw=1, alpha=0.6)
ax.axvline(dates[INTERVENTION_PERIOD], color=MOZILLA_DARK, ls="--", lw=1.5, label="Intervention")
ax.set_title("Indexed Firefox Share (Pre-Period = 100)\nParallel Trends Check", fontsize=12, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Indexed Share")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("charts/01_parallel_trends.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: charts/01_parallel_trends.png")

# Pre-period correlation test (treated vs avg control)
avg_control_pre = (
    panel[(panel["is_treated"] == False) & (panel["is_post"] == False)]
    .groupby("period")["firefox_share"].mean()
)
for geo in TREATED_GEOS:
    treated_pre = (
        panel[(panel["geo"] == geo) & (panel["is_post"] == False)]
        .set_index("period")["firefox_share"]
    )
    corr, p = stats.pearsonr(treated_pre, avg_control_pre)
    print(f"  Pre-period correlation {geo} vs avg-control: r={corr:.3f}, p={p:.4f}")

# ─────────────────────────────────────────────────────────────
# 3. SYNTHETIC CONTROL CONSTRUCTION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Synthetic Control construction")
print("=" * 60)

def build_synthetic_control(treated_geo, panel, donor_geos, pre_periods):
    """
    Fit synthetic control: find convex combination of donors
    that minimizes pre-period MSE vs treated geo.
    Returns weights and full synthetic control series.
    """
    pre_mask = panel["period"] < pre_periods
    treated_pre = (
        panel[(panel["geo"] == treated_geo) & pre_mask]
        .sort_values("period")["firefox_share"].values
    )
    donor_matrix = np.column_stack([
        panel[(panel["geo"] == geo) & pre_mask]
        .sort_values("period")["firefox_share"].values
        for geo in donor_geos
    ])
    
    n_donors = len(donor_geos)
    
    def objective(w):
        synth = donor_matrix @ w
        return np.sum((treated_pre - synth) ** 2)
    
    # Constraints: weights sum to 1, all non-negative
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n_donors
    w0 = np.ones(n_donors) / n_donors
    
    result = opt.minimize(
        objective, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12}
    )
    
    weights = result.x
    
    # Build full synthetic control (pre + post)
    full_donor_matrix = np.column_stack([
        panel[panel["geo"] == geo]
        .sort_values("period")["firefox_share"].values
        for geo in donor_geos
    ])
    synthetic_series = full_donor_matrix @ weights
    
    pre_mse  = result.fun / pre_periods
    pre_rmse = np.sqrt(pre_mse)
    
    return weights, synthetic_series, pre_rmse

SC_RESULTS = {}

for treated_geo in TREATED_GEOS:
    donor_geos = CONTROL_GEOS
    weights, synthetic, pre_rmse = build_synthetic_control(
        treated_geo, panel, donor_geos, INTERVENTION_PERIOD
    )
    
    # Compute pointwise lift
    actual = panel[panel["geo"] == treated_geo].sort_values("period")["firefox_share"].values
    lift    = actual - synthetic
    post_lift = lift[INTERVENTION_PERIOD:]
    
    # Mean post-period lift
    mean_lift  = post_lift.mean()
    cum_lift   = post_lift.sum()
    
    SC_RESULTS[treated_geo] = {
        "weights": dict(zip(donor_geos, weights.round(4))),
        "synthetic": synthetic,
        "actual": actual,
        "lift": lift,
        "post_lift": post_lift,
        "mean_lift_pp": mean_lift,
        "cum_lift_pp": cum_lift,
        "pre_rmse": pre_rmse,
    }
    
    print(f"\n  {treated_geo}:")
    print(f"    Pre-period RMSE: {pre_rmse:.4f} pp")
    print(f"    Mean post-period lift: {mean_lift:.3f} pp")
    print(f"    Cumulative lift: {cum_lift:.2f} pp-months")
    top3_donors = sorted(
        SC_RESULTS[treated_geo]["weights"].items(), key=lambda x: -x[1]
    )[:3]
    print(f"    Top donors: {[(g, round(w,3)) for g,w in top3_donors]}")

# ─────────────────────────────────────────────────────────────
# 4. SYNTHETIC CONTROL CHART (Actual vs Counterfactual)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Synthetic control overlay charts")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
COLORS = {"DE": MOZILLA_RED, "FR": MOZILLA_BLUE, "PL": "#F59E0B"}

for ax, geo in zip(axes, TREATED_GEOS):
    res = SC_RESULTS[geo]
    ax.plot(dates, res["actual"], color=COLORS[geo], lw=2.5, label="Actual (treated)")
    ax.plot(dates, res["synthetic"], color=MOZILLA_DARK, lw=2.0, ls="--", label="Synthetic control")
    ax.axvline(dates[INTERVENTION_PERIOD], color=MOZILLA_GRAY, ls=":", lw=1.8, label="Campaign start")
    ax.fill_between(
        dates[INTERVENTION_PERIOD:],
        res["synthetic"][INTERVENTION_PERIOD:],
        res["actual"][INTERVENTION_PERIOD:],
        alpha=0.18, color=COLORS[geo], label="Estimated lift"
    )
    ax.set_title(f"{geo} — Synthetic Control\nMean lift: {res['mean_lift_pp']:.2f} pp",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Firefox Share (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

plt.suptitle("Synthetic Control: Actual vs Counterfactual Firefox Share", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("charts/02_synthetic_control.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: charts/02_synthetic_control.png")

# ─────────────────────────────────────────────────────────────
# 5. BOOTSTRAPPED CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Bootstrapped confidence intervals on lift")
print("=" * 60)

N_BOOTSTRAP = 500
CI_RESULTS = {}

for geo in TREATED_GEOS:
    res = SC_RESULTS[geo]
    donor_geos = CONTROL_GEOS
    
    # Pre-period RMSE as proxy for noise; bootstrap by resampling donors
    boot_mean_lifts = []
    boot_rng = np.random.default_rng(RNG_SEED + 7777)
    
    for _ in range(N_BOOTSTRAP):
        # Resample donors with replacement
        boot_donors = boot_rng.choice(donor_geos, size=len(donor_geos), replace=True)
        unique_donors = list(dict.fromkeys(boot_donors))  # preserve order, deduplicate
        if len(unique_donors) < 2:
            unique_donors = donor_geos[:3]
        
        try:
            _, synth_boot, _ = build_synthetic_control(
                geo, panel, unique_donors, INTERVENTION_PERIOD
            )
            actual_vals = res["actual"]
            boot_lift   = actual_vals[INTERVENTION_PERIOD:] - synth_boot[INTERVENTION_PERIOD:]
            boot_mean_lifts.append(boot_lift.mean())
        except Exception:
            boot_mean_lifts.append(res["mean_lift_pp"])
    
    ci_lo = np.percentile(boot_mean_lifts, 2.5)
    ci_hi = np.percentile(boot_mean_lifts, 97.5)
    CI_RESULTS[geo] = {
        "mean_lift": res["mean_lift_pp"],
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "boot_lifts": boot_mean_lifts,
    }
    print(f"  {geo}: lift = {res['mean_lift_pp']:.3f} pp  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

# Cumulative lift chart
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for ax, geo in zip(axes, TREATED_GEOS):
    res = SC_RESULTS[geo]
    cumulative = np.cumsum(res["post_lift"])
    post_dates  = dates[INTERVENTION_PERIOD:]
    ax.plot(post_dates, cumulative, color=COLORS[geo], lw=2.5)
    ax.fill_between(post_dates, 0, cumulative, alpha=0.15, color=COLORS[geo])
    ax.axhline(0, color=MOZILLA_DARK, lw=0.8, ls=":")
    ci = CI_RESULTS[geo]
    ax.set_title(
        f"{geo} — Cumulative Lift\n"
        f"Mean: {ci['mean_lift']:.2f} pp  95%CI [{ci['ci_lo']:.2f}, {ci['ci_hi']:.2f}]",
        fontsize=10, fontweight="bold"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Lift (pp-months)")
    ax.grid(True, alpha=0.4)

plt.suptitle("Cumulative Incremental Lift Post-Campaign Launch", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("charts/03_cumulative_lift.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: charts/03_cumulative_lift.png")

# ─────────────────────────────────────────────────────────────
# 6. PLACEBO / PERMUTATION TESTS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Placebo / permutation tests")
print("=" * 60)

PLACEBO_RESULTS = {}

for geo in ALL_GEOS:
    donor_pool = [g for g in ALL_GEOS if g != geo]
    _, synth, pre_rmse = build_synthetic_control(geo, panel, donor_pool, INTERVENTION_PERIOD)
    actual = panel[panel["geo"] == geo].sort_values("period")["firefox_share"].values
    post_lift = (actual - synth)[INTERVENTION_PERIOD:]
    PLACEBO_RESULTS[geo] = {
        "mean_post_lift": post_lift.mean(),
        "pre_rmse": pre_rmse,
    }

# Empirical p-values: proportion of placebo geos with |lift| >= observed lift
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
pvals = {}

for ax, geo in zip(axes, TREATED_GEOS):
    observed_lift = PLACEBO_RESULTS[geo]["mean_post_lift"]
    control_lifts = [PLACEBO_RESULTS[g]["mean_post_lift"] for g in CONTROL_GEOS]
    all_placebo   = control_lifts  # 12 controls
    
    # p-value: fraction of placebos with lift >= observed
    pval = np.mean([abs(l) >= abs(observed_lift) for l in all_placebo])
    pvals[geo] = pval
    
    print(f"  {geo}: observed lift={observed_lift:.3f} pp, empirical p={pval:.3f}")
    
    # Distribution plot
    ax.hist(all_placebo, bins=8, color=MOZILLA_LIGHT,
            edgecolor=MOZILLA_DARK, alpha=0.85, label="Placebo controls")
    ax.axvline(observed_lift, color=COLORS[geo], lw=2.5, ls="--",
               label=f"Treated ({geo}): {observed_lift:.2f} pp")
    ax.set_title(f"{geo} — Placebo Distribution\np={pval:.3f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Post-Period Lift (pp)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

plt.suptitle("Placebo Tests: Treated Geo Lift vs Control Geo Null Distribution",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("charts/04_placebo_tests.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: charts/04_placebo_tests.png")

# ─────────────────────────────────────────────────────────────
# 7. XGBoost + SHAP — What drives lift?
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: XGBoost + SHAP — geo-level lift drivers")
print("=" * 60)

# Build a geo × period feature table for post-intervention period
# Target: actual firefox_share in post period
# Features: geo characteristics + campaign spend + pre-period baseline

post_panel = panel[panel["is_post"] == True].copy()

# Add synthetic control counterfactual as a feature
synth_rows = []
for geo in ALL_GEOS:
    donor_pool = [g for g in ALL_GEOS if g != geo]
    _, synth, _ = build_synthetic_control(geo, panel, donor_pool, INTERVENTION_PERIOD)
    for t_idx in range(INTERVENTION_PERIOD, N_PERIODS):
        synth_rows.append({
            "geo": geo,
            "period": t_idx,
            "synth_share": synth[t_idx],
        })

synth_df = pd.DataFrame(synth_rows)
post_panel = post_panel.merge(synth_df, on=["geo", "period"], how="left")
post_panel["lift_pp"] = post_panel["firefox_share"] - post_panel["synth_share"]

# Pre-period baseline (last 6 months of pre)
pre_baseline = (
    panel[(panel["is_post"] == False) & (panel["period"] >= INTERVENTION_PERIOD - 6)]
    .groupby("geo")["firefox_share"].mean()
    .rename("pre_baseline")
    .reset_index()
)
post_panel = post_panel.merge(pre_baseline, on="geo", how="left")

# Months since intervention
post_panel["months_since_start"] = post_panel["period"] - INTERVENTION_PERIOD

FEATURE_COLS = [
    "pop_mm", "broadband_pct", "tech_index",
    "campaign_spend_usd", "pre_baseline",
    "months_since_start", "is_treated"
]

X = post_panel[FEATURE_COLS].astype(float)
y = post_panel["firefox_share"].values

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RNG_SEED,
    verbosity=0
)
model.fit(X, y)

y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2   = r2_score(y, y_pred)
print(f"  XGBoost RMSE: {rmse:.4f} pp")
print(f"  XGBoost R²  : {r2:.4f}")

# SHAP
explainer  = shap.TreeExplainer(model)
shap_vals  = explainer.shap_values(X)
shap_df    = pd.DataFrame(shap_vals, columns=FEATURE_COLS)

# Mean absolute SHAP per feature
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
print("\n  Mean |SHAP| by feature:")
for feat, val in mean_abs_shap.items():
    print(f"    {feat:25s}: {val:.4f}")

# SHAP bar chart
fig, ax = plt.subplots(figsize=(9, 5))
colors_bar = [MOZILLA_RED if v > 0 else MOZILLA_BLUE
              for v in shap_df.mean().reindex(mean_abs_shap.index)]
bars = ax.barh(
    mean_abs_shap.index[::-1],
    mean_abs_shap.values[::-1],
    color=[MOZILLA_RED if c == MOZILLA_RED else MOZILLA_BLUE
           for c in colors_bar[::-1]],
    edgecolor="white", linewidth=0.5
)
ax.set_xlabel("Mean |SHAP Value| (impact on Firefox share prediction)")
ax.set_title("SHAP Feature Importance\nXGBoost — Post-Campaign Firefox Share Predictor",
             fontsize=12, fontweight="bold")
ax.grid(True, axis="x", alpha=0.4)

# Nicer labels
LABEL_MAP = {
    "campaign_spend_usd": "Campaign Spend ($)",
    "pre_baseline": "Pre-campaign Baseline Share",
    "tech_index": "Tech-Savviness Index",
    "broadband_pct": "Broadband Penetration (%)",
    "months_since_start": "Months Since Campaign Start",
    "pop_mm": "Population (MM)",
    "is_treated": "Is Treated Geo",
}
ax.set_yticklabels([LABEL_MAP.get(l.get_text(), l.get_text())
                    for l in ax.get_yticklabels()])
plt.tight_layout()
plt.savefig("charts/05_shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: charts/05_shap_importance.png")

# ─────────────────────────────────────────────────────────────
# 8. ROI ESTIMATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: ROI Estimation")
print("=" * 60)

# Public Firefox MAU: ~180M globally (2024).
# Approx EU share ~22% → ~40M EU MAU.
# Each geo's share of EU internet users (Eurostat proxy).
GEO_INTERNET_USERS_MM = {
    "DE": 72.6, "FR": 57.0, "PL": 30.5
}
FIREFOX_GLOBAL_MAU      = 180_000_000
EUR_INTERNET_USERS_MM   = 350.0  # EU + UK approx

roi_rows = []
for geo in TREATED_GEOS:
    res     = SC_RESULTS[geo]
    ci      = CI_RESULTS[geo]
    lift_pp = res["mean_lift_pp"]    # percentage points
    
    internet_users = GEO_INTERNET_USERS_MM[geo] * 1_000_000
    incremental_installs = (lift_pp / 100) * internet_users
    total_spend = panel[(panel["geo"] == geo) & panel["is_post"]]["campaign_spend_usd"].sum()
    
    cpi = total_spend / incremental_installs if incremental_installs > 0 else float("inf")
    
    roi_rows.append({
        "Geo": geo,
        "Lift (pp)": round(lift_pp, 3),
        "95% CI Lo": round(ci["ci_lo"], 3),
        "95% CI Hi": round(ci["ci_hi"], 3),
        "True Lift (pp)": TRUE_LIFT[geo],
        "Incremental Installs (est.)": int(incremental_installs),
        "Total Campaign Spend ($)": int(total_spend),
        "Cost per Install ($)": round(cpi, 2),
        "p-value": round(pvals[geo], 3),
    })
    print(f"  {geo}: ~{incremental_installs:,.0f} incremental installs "
          f"@ ${total_spend:,.0f} spend → CPI ${cpi:.2f}")

roi_df = pd.DataFrame(roi_rows)
print("\n  ROI Summary:")
print(roi_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 9. SUMMARY DASHBOARD CHART
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Summary dashboard chart")
print("=" * 60)

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

# Row 0: lift estimates with CI
ax0 = fig.add_subplot(gs[0, :2])
geos_plot = TREATED_GEOS
lifts   = [CI_RESULTS[g]["mean_lift"] for g in geos_plot]
ci_los  = [CI_RESULTS[g]["ci_lo"]     for g in geos_plot]
ci_his  = [CI_RESULTS[g]["ci_hi"]     for g in geos_plot]
true_ls = [TRUE_LIFT[g]               for g in geos_plot]

x = np.arange(len(geos_plot))
bar_colors = [MOZILLA_RED, MOZILLA_BLUE, "#F59E0B"]
bars = ax0.bar(x, lifts, width=0.5, color=bar_colors, alpha=0.85, zorder=3)
ax0.errorbar(x, lifts,
             yerr=[np.array(lifts) - np.array(ci_los), np.array(ci_his) - np.array(lifts)],
             fmt="none", ecolor=MOZILLA_DARK, elinewidth=2, capsize=6, zorder=4)
ax0.scatter(x, true_ls, marker="D", s=80, color=MOZILLA_DARK, zorder=5, label="True lift (synthetic ground truth)")
ax0.set_xticks(x)
ax0.set_xticklabels(geos_plot, fontsize=12)
ax0.set_ylabel("Mean Lift (percentage points)")
ax0.set_title("Estimated vs True Lift by Geo\n(bars = estimated, diamonds = ground truth)",
              fontweight="bold")
ax0.legend()
ax0.grid(True, axis="y", alpha=0.4)
for i, (l, t) in enumerate(zip(lifts, true_ls)):
    ax0.text(i, l + 0.05, f"{l:.2f}pp", ha="center", fontsize=10, color=bar_colors[i])

# Row 0, col 2: CPI bar
ax1 = fig.add_subplot(gs[0, 2])
cpis = [row["Cost per Install ($)"] for row in roi_rows]
ax1.bar(geos_plot, cpis, color=bar_colors, alpha=0.85)
ax1.set_ylabel("Cost per Incremental Install ($)")
ax1.set_title("Estimated CPI\nby Geo", fontweight="bold")
ax1.grid(True, axis="y", alpha=0.4)
for i, c in enumerate(cpis):
    ax1.text(i, c + 0.02, f"${c:.2f}", ha="center", fontsize=10)

# Row 1: placebo lift distributions
ax2 = fig.add_subplot(gs[1, :])
for i, geo in enumerate(TREATED_GEOS):
    observed_lift = PLACEBO_RESULTS[geo]["mean_post_lift"]
    control_lifts = [PLACEBO_RESULTS[g]["mean_post_lift"] for g in CONTROL_GEOS]
    offset = i * 0.25
    ax2.hist(
        [l + offset for l in control_lifts],
        bins=8, alpha=0.45, color=bar_colors[i],
        label=f"{geo} placebo pool", density=True
    )
    ax2.axvline(observed_lift + offset, color=bar_colors[i], lw=2.5, ls="--",
                label=f"{geo} observed ({observed_lift:.2f}pp)")
ax2.set_xlabel("Mean Post-Period Lift (pp) [offset for visibility]")
ax2.set_title("Placebo Distribution: Treated Lift vs Control Null Distribution",
              fontweight="bold")
ax2.legend(ncol=3, fontsize=9)
ax2.grid(True, axis="x", alpha=0.4)

plt.suptitle("Geo Marketing Lift — Synthetic Control Summary Dashboard",
             fontsize=14, fontweight="bold", y=1.01)
plt.savefig("charts/06_summary_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: charts/06_summary_dashboard.png")

# ─────────────────────────────────────────────────────────────
# 10. SAVE RESULTS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Saving results to JSON for RESULTS.md generation")
print("=" * 60)

final_results = {
    "geos": {},
    "xgboost": {"rmse": round(rmse, 4), "r2": round(r2, 4)},
    "shap_top_features": mean_abs_shap.index.tolist()[:3],
}

for geo in TREATED_GEOS:
    final_results["geos"][geo] = {
        "estimated_lift_pp": round(SC_RESULTS[geo]["mean_lift_pp"], 3),
        "true_lift_pp": TRUE_LIFT[geo],
        "ci_lo": round(CI_RESULTS[geo]["ci_lo"], 3),
        "ci_hi": round(CI_RESULTS[geo]["ci_hi"], 3),
        "p_value": round(pvals[geo], 3),
        "pre_rmse": round(SC_RESULTS[geo]["pre_rmse"], 4),
    }

for row in roi_rows:
    geo = row["Geo"]
    final_results["geos"][geo]["incremental_installs"] = row["Incremental Installs (est.)"]
    final_results["geos"][geo]["total_spend"] = row["Total Campaign Spend ($)"]
    final_results["geos"][geo]["cpi"] = row["Cost per Install ($)"]

with open("results.json", "w") as f:
    json.dump(final_results, f, indent=2)

print("  Saved: results.json")
print("\n✅ Analysis complete. All charts saved to charts/")
print("\n" + "=" * 60)
print("FINAL METRICS SUMMARY")
print("=" * 60)
print(f"{'Geo':<6} {'Est.Lift(pp)':<14} {'True(pp)':<12} {'95%CI':<22} {'p-val':<8} {'CPI($)'}")
print("-" * 80)
for row in roi_rows:
    g = row["Geo"]
    r = final_results["geos"][g]
    print(f"{g:<6} {r['estimated_lift_pp']:<14.3f} {r['true_lift_pp']:<12.1f} "
          f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]   {r['p_value']:<8.3f} {r['cpi']:.2f}")
print(f"\nXGBoost companion model: RMSE={rmse:.4f} pp, R²={r2:.4f}")
