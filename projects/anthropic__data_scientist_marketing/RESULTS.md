# Results — Lifecycle Segment Lift

All metrics produced by a single clean run of `analysis.py` on 5,000 synthetic users (seed=42).

---

## Dataset

| Stat | Value |
|---|---|
| Total users | 5,000 |
| Overall conversion rate | 33.7% |
| Email treatment rate | 58.8% |

---

## Causal Lift Estimation (Propensity Score Matching)

| Method | ATE (Δ conversion rate) | Notes |
|---|---|---|
| Naive ATE (raw difference) | **+0.1591** | Biased upward — treated users are inherently more active |
| PSM-Matched ATE (causal) | **+0.0768** | Apples-to-apples after balancing on engagement |
| Confounding bias removed | **0.0823 pp** | ~52% of naive effect was selection bias |
| Propensity model AUROC | 0.639 | Predicts who gets the email |
| Sessions imbalance (before) | 2.442 | Raw difference in avg sessions, treated vs. control |
| Sessions imbalance (after) | 0.123 | 95% reduction in covariate imbalance post-match |

---

## User Segments (KMeans, k=4)

| Segment | n | Avg Sessions | Avg Messages | Avg Tenure (days) | Conversion Rate |
|---|---|---|---|---|---|
| Low-Activity New | 1,566 | 6.5 | 63.5 | 69 | 19.8% |
| Moderate Users | 1,627 | 6.9 | 64.4 | 20 | 29.3% |
| Engaged Tenured | 891 | 7.9 | 132.4 | 47 | 35.6% |
| Power Users | 916 | 18.6 | 70.3 | 46 | 63.6% |

---

## Conversion Prediction (XGBoost on Matched Sample, 5-fold CV)

| Metric | Value |
|---|---|
| AUROC | **0.798** |
| AUPRC | **0.713** |
| Accuracy | 75.3% |
| F1 (converted class) | 0.62 |

---

## SHAP Feature Importance (Mean |SHAP|)

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | Sessions (14d) | 0.805 |
| 2 | Plan: Pro | 0.326 |
| 3 | Days Since Signup | 0.253 |
| 4 | Messages Sent | 0.246 |
| 5 | Email Received | 0.199 |
| 6 | Industry: Tech | 0.117 |

---

## Predicted Email Lift by Segment

| Segment | Avg Predicted Lift |
|---|---|
| Power Users | +0.092 |
| Engaged Tenured | +0.067 |
| Moderate Users | +0.066 |
| Low-Activity New | +0.059 |

---

## Charts

| File | Description |
|---|---|
| `charts/eda_conversion_by_segment.png` | Conversion rates by plan and industry |
| `charts/eda_correlation_heatmap.png` | Feature correlation matrix |
| `charts/eda_feature_distributions.png` | Distribution of key features by conversion status |
| `charts/segmentation_elbow.png` | KMeans elbow curve (k=2..8) |
| `charts/segmentation_heatmap.png` | Cluster profile heatmap (normalized + annotated) |
| `charts/causal_ate_comparison.png` | Naive vs. matched ATE bar chart |
| `charts/propensity_score_overlap.png` | Propensity score overlap before and after matching |
| `charts/model_roc_pr_curves.png` | ROC and Precision-Recall curves |
| `charts/shap_beeswarm.png` | SHAP beeswarm plot |
| `charts/shap_importance_bar.png` | Mean absolute SHAP feature importance |
| `charts/shap_dependence.png` | SHAP dependence plots (sessions, messages) |
| `charts/lift_by_segment.png` | Predicted email lift by user segment |
