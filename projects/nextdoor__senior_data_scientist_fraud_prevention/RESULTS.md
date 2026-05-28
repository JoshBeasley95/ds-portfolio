# Results — Neighborhood Trust Score: Fraud Detection

> All metrics produced by a single end-to-end run of `analysis.py` on 2026-05-28.
> Dataset: **OpenML Credit Card Fraud (ULB, public domain)** — used as a synthetic proxy for
> account-registration fraud signals. NOT Nextdoor internal data.

---

## Model Performance (XGBoost + SMOTE, test set n=10,099)

| Metric | Value |
|--------|-------|
| **ROC-AUC** | **0.9818** |
| **PR-AUC (Average Precision)** | **0.9223** |
| **Best F1-Score** | **0.9206** |
| Optimal Decision Threshold | 0.827 |
| Precision @ threshold | **0.9560** |
| Recall @ threshold | **0.8878** |
| True Positives (fraud caught) | 87 |
| False Positives (false alarms) | 4 |
| False Negatives (missed fraud) | 11 |
| True Negatives (legit passed) | 9,997 |

## Dataset Summary

| Item | Value |
|------|-------|
| Source | OpenML Credit Card Fraud (ULB, public domain) |
| Total rows (working subset) | 50,492 |
| Fraud events | 492 (0.97%) |
| Legitimate events | 50,000 |
| Train split (pre-SMOTE) | 40,393 rows, 394 fraud |
| Train split (post-SMOTE) | 79,998 rows, 39,999 fraud |
| Test split | 10,099 rows, 98 fraud |
| Features engineered | 33 (28 V-features + log_amount, sin_hour, cos_hour, amount_bucket, amount_zscore) |

## Top SHAP Feature

| Feature | Mean \|SHAP\| |
|---------|-------------|
| `v14` | highest mean absolute SHAP value (top fraud driver) |

## Classification Report

```
              precision    recall  f1-score   support
  Legitimate       1.00      1.00      1.00     10001
       Fraud       0.96      0.89      0.92        98
    accuracy                           1.00     10099
   macro avg       0.98      0.94      0.96     10099
weighted avg       1.00      1.00      1.00     10099
```

---

## Charts Produced

| File | Description |
|------|-------------|
| `charts/class_imbalance.png` | Class distribution + log(Amount) distributions by label |
| `charts/pr_roc_curves.png` | ROC curve (AUC=0.9818) + Precision-Recall curve (AUCPR=0.9223) |
| `charts/confusion_matrix.png` | Confusion matrix at optimal threshold (0.827) |
| `charts/shap_summary.png` | SHAP beeswarm plot — top 15 global fraud drivers |
| `charts/shap_waterfall_example.png` | SHAP waterfall — highest-risk account in test set |
| `charts/shap_importance_bar.png` | Mean \|SHAP\| bar chart for top 15 features |
