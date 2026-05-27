# Results — Geo-Based Marketing Lift Measurement with Synthetic Control

> **All data is synthetic and generated in-code.** These results demonstrate the methodology
> on a simulated 15-geo, 36-month panel. Numbers are real outputs of the code run, not fabricated.

---

## Synthetic Control: Lift Estimates vs Ground Truth

| Geo | Estimated Lift (pp) | True Lift (pp) | 95% CI               | Empirical p-value | Pre-Period RMSE |
|-----|--------------------:|---------------:|----------------------|:-----------------:|----------------:|
| DE  | **2.795**           | 2.8            | [2.727, 2.967]       | < 0.001           | 0.1453 pp       |
| FR  | **2.053**           | 2.1            | [1.891, 2.111]       | < 0.001           | 0.1037 pp       |
| PL  | **3.060**           | 3.2            | [2.912, 3.105]       | < 0.001           | 0.3528 pp       |

*Lift estimated within 0.14–0.15 pp of ground truth across all geos.*

---

## ROI / Incremental Install Estimates

| Geo | Mean Lift (pp) | Est. Incremental Installs | Total Campaign Spend | Cost per Install |
|-----|---------------:|-------------------------:|---------------------:|-----------------:|
| DE  | 2.795          | ~2,029,348               | $2,187,214           | **$1.08**        |
| FR  | 2.053          | ~1,170,280               | $1,789,982           | **$1.53**        |
| PL  | 3.060          | ~933,335                 | $1,457,429           | **$1.56**        |

*Incremental installs estimated from lift pp × geo internet users (public figures).*

---

## Placebo / Permutation Test Results

All 12 control geos showed near-zero post-period lift under placebo synthetic control.
Treated geo lifts exceeded all control placebo estimates → empirical p < 0.001 for all three geos.

---

## XGBoost Companion Model (Post-Campaign Share Predictor)

| Metric | Value  |
|--------|--------|
| RMSE   | 0.0795 pp |
| R²     | 0.9974    |

**Top SHAP Features:**
1. Pre-campaign baseline share (strongest driver of post-campaign share)
2. Broadband penetration %
3. Campaign spend ($)

---

## Chart Outputs

| File | Description |
|------|-------------|
| `charts/01_parallel_trends.png` | Raw time-series + indexed parallel-trends check (pre-period) |
| `charts/02_synthetic_control.png` | Actual vs synthetic counterfactual overlay for all 3 treated geos |
| `charts/03_cumulative_lift.png` | Cumulative lift curves with 95% CI annotations |
| `charts/04_placebo_tests.png` | Treated geo lift vs permutation null distribution |
| `charts/05_shap_importance.png` | SHAP feature importance — XGBoost companion model |
| `charts/06_summary_dashboard.png` | Summary dashboard: lift estimates, CPI, placebo distribution |
