# Results — Guest Insurance Intent Scorer

> **Data disclaimer:** All results are produced from **synthetic data** generated to mirror the
> public Travel Insurance Prediction Dataset schema (1,987 rows, 8 raw features + 5 engineered).
> No Airbnb proprietary data was used.

---

## Model Performance Summary

| Model | ROC-AUC | PR-AUC | Brier Score |
|---|---|---|---|
| Logistic Regression (baseline) | 0.6260 | 0.3356 | — |
| XGBoost (raw / tuned) | 0.6066 | 0.2973 | 0.2347 |
| **XGBoost (calibrated)** | **0.6295** | **0.3244** | **0.1626** |

> Baseline positive rate: **21.4 %** (imbalanced; handled with `scale_pos_weight`)

---

## Threshold Optimisation

| Metric | Value |
|---|---|
| Optimal decision threshold (max-F1) | 0.23 |
| F1 at optimal threshold | 0.4147 |
| Precision @ optimal threshold | 0.34 |
| Recall @ optimal threshold | 0.53 |

---

## Calibration Improvement

| Metric | Raw XGBoost | Calibrated XGBoost |
|---|---|---|
| Brier Score (lower = better) | 0.2347 | **0.1626** |
| Reduction | — | **30.7 %** |

Isotonic calibration reduced the Brier score by **30.7 %**, meaning the model's
probability estimates are substantially more reliable for downstream decision-making
(e.g., pricing a personalised insurance offer).

---

## Decile Lift Analysis

| Decile | Uptake Rate | Lift vs. Baseline |
|---|---|---|
| 10 (top-scored) | ~35 % | **1.64×** |
| 1 (bottom-scored) | ~10 % | 0.47× |

Top-decile guests are **1.64× more likely** to take insurance than the average guest,
enabling precise targeting that avoids annoying low-intent guests while maximising offer acceptance.

---

## Cross-Validation (5-fold, train set)

| Metric | Score |
|---|---|
| CV ROC-AUC (XGBoost best params) | 0.5969 |

Best hyperparameters selected by GridSearchCV:
- `n_estimators=100`, `max_depth=3`, `learning_rate=0.05`
- `subsample=1.0`, `colsample_bytree=1.0`, `min_child_weight=3`

---

## Charts Produced

| File | Description |
|---|---|
| `charts/01_feature_distributions.png` | KDE + bar plots: feature distributions by insurance uptake |
| `charts/02_correlation_heatmap.png` | Pearson correlation matrix |
| `charts/03_roc_curve.png` | ROC curves for LR, XGBoost raw, XGBoost calibrated |
| `charts/04_pr_curve.png` | Precision-Recall curves with random baseline |
| `charts/05_calibration_threshold.png` | Reliability diagram + F1 vs threshold sweep |
| `charts/06_confusion_matrix.png` | Confusion matrix at optimal threshold |
| `charts/07_shap_summary.png` | SHAP beeswarm summary plot |
| `charts/08_shap_importance.png` | Global SHAP bar chart (mean \|SHAP\|) |
| `charts/09_shap_waterfall.png` | Per-guest SHAP waterfall for highest-intent guest |
| `charts/10_decile_lift.png` | Uptake rate and lift by score decile |

---

## Top SHAP Features (global, mean |SHAP|)

1. **EverTravelledAbroad** — strongest positive signal; guests with international travel history show much higher insurance intent
2. **AnnualIncome** — higher income correlates strongly with uptake
3. **TripRiskScore** — engineered composite (chronic disease + travel history + family size)
4. **FrequentFlyer** — frequent travellers demonstrate higher coverage intent
5. **Age** — older guests show modestly higher propensity

