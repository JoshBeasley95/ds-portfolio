"""
Guest Insurance Intent Scorer
Airbnb × Senior Data Scientist, Guest Travel Insurance (Algorithms)
──────────────────────────────────────────────────────────────────
Demonstration on PUBLIC / SYNTHETIC data that mirrors the Travel Insurance
Prediction Dataset schema (1,987 rows).  No Airbnb proprietary data is used.

Pipeline: synthetic data generation → EDA → feature engineering →
          baseline logistic regression → XGBoost + tuning →
          ROC/PR/calibration evaluation → SHAP explainability →
          threshold optimisation → chart export
"""

import warnings, os, json, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    brier_score_loss, f1_score
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap
import joblib

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

os.makedirs("charts", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 0.  Palette / style
# ─────────────────────────────────────────────────────────────
AIRBNB_RED   = "#FF5A5F"
AIRBNB_TEAL  = "#00A699"
AIRBNB_GRAY  = "#767676"
AIRBNB_LIGHT = "#F7F7F7"

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": plt.cycler(color=[AIRBNB_RED, AIRBNB_TEAL,
                                         "#FC642D", "#484848", AIRBNB_GRAY]),
})

# ─────────────────────────────────────────────────────────────
# 1.  Synthetic data generation (mirrors real dataset schema)
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("1. Generating synthetic dataset …")

N = 1987

age             = np.clip(np.random.normal(30, 8, N).astype(int), 18, 65)
employment_type = np.random.choice(
    ["Government Sector", "Private Sector/Self Employed"],
    p=[0.27, 0.73], size=N)
graduate        = np.random.choice(["Yes", "No"], p=[0.87, 0.13], size=N)
annual_income   = np.clip(
    np.random.lognormal(mean=11.3, sigma=0.55, size=N).astype(int),
    300_000, 1_800_000)
family_members  = np.random.choice(range(2, 10), p=[0.22,0.24,0.22,0.14,0.09,0.05,0.03,0.01], size=N)
chronic_disease = np.random.choice([0, 1], p=[0.72, 0.28], size=N)
frequent_flyer  = np.random.choice(["Yes", "No"], p=[0.21, 0.79], size=N)
ever_abroad     = np.random.choice(["Yes", "No"], p=[0.24, 0.76], size=N)

# Construct a realistic propensity
logit = (
    -2.2
    + 0.025 * (age - 30)
    + 0.6  * (employment_type == "Government Sector")
    + 0.4  * (ever_abroad == "Yes")
    + 0.5  * (frequent_flyer == "Yes")
    + 0.3  * (chronic_disease == 1)
    + 0.002 * (annual_income / 50_000)
    + 0.15 * (family_members - 4)
    + 0.25 * (graduate == "Yes")
    + np.random.normal(0, 0.8, N)
)
p_insure        = 1 / (1 + np.exp(-logit))
travel_insurance = (np.random.uniform(size=N) < p_insure).astype(int)

df = pd.DataFrame({
    "Age":                  age,
    "EmploymentType":       employment_type,
    "GraduateOrNot":        graduate,
    "AnnualIncome":         annual_income,
    "FamilyMembers":        family_members,
    "ChronicDiseases":      chronic_disease,
    "FrequentFlyer":        frequent_flyer,
    "EverTravelledAbroad":  ever_abroad,
    "TravelInsurance":      travel_insurance,
})

print(f"   Dataset shape : {df.shape}")
print(f"   Class balance : {df.TravelInsurance.value_counts().to_dict()}")
print(f"   Positive rate : {df.TravelInsurance.mean():.3f}")

# ─────────────────────────────────────────────────────────────
# 2.  EDA
# ─────────────────────────────────────────────────────────────
print("\n2. Exploratory Data Analysis …")

# --- 2a. Feature distribution grid ---
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
axes = axes.ravel()

num_cols = ["Age", "AnnualIncome", "FamilyMembers"]
cat_cols = ["EmploymentType", "GraduateOrNot", "FrequentFlyer",
            "EverTravelledAbroad", "ChronicDiseases"]

for i, col in enumerate(num_cols):
    for label, color in [(0, AIRBNB_GRAY), (1, AIRBNB_RED)]:
        sns.kdeplot(df[df.TravelInsurance == label][col],
                    ax=axes[i], fill=True, alpha=0.4, color=color,
                    label=f"Insurance={'Yes' if label else 'No'}")
    axes[i].set_title(col, fontweight="bold")
    axes[i].legend(fontsize=9)

for j, col in enumerate(cat_cols):
    ax = axes[j + 3]
    prop = (df.groupby(col)["TravelInsurance"]
              .mean()
              .reset_index()
              .rename(columns={"TravelInsurance": "Uptake Rate"}))
    sns.barplot(data=prop, x=col, y="Uptake Rate", ax=ax,
                palette=[AIRBNB_RED, AIRBNB_TEAL],
                order=prop.sort_values("Uptake Rate", ascending=False)[col])
    ax.set_title(col, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width()/2, p.get_height() + 0.02),
                    ha="center", fontsize=9)
    ax.tick_params(axis="x", labelsize=8, rotation=15)

plt.suptitle("Feature Distributions by Insurance Uptake", fontsize=14, y=1.01, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/01_feature_distributions.png", bbox_inches="tight")
plt.close()
print("   Saved charts/01_feature_distributions.png")

# --- 2b. Correlation heatmap (numeric + encoded) ---
df_enc = df.copy()
bool_maps = {
    "GraduateOrNot":       {"Yes": 1, "No": 0},
    "FrequentFlyer":       {"Yes": 1, "No": 0},
    "EverTravelledAbroad": {"Yes": 1, "No": 0},
    "EmploymentType":      {"Government Sector": 1, "Private Sector/Self Employed": 0},
}
for col, mapping in bool_maps.items():
    df_enc[col] = df_enc[col].map(mapping)

corr = df_enc.select_dtypes(include="number").corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Pearson Correlation Matrix (numeric + binary-encoded features)",
             fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig("charts/02_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("   Saved charts/02_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────
# 3.  Feature Engineering
# ─────────────────────────────────────────────────────────────
print("\n3. Feature Engineering …")

df_fe = df_enc.copy()

# Income buckets (INR)
df_fe["IncomeBucket"] = pd.cut(
    df_fe["AnnualIncome"],
    bins=[0, 400_000, 700_000, 1_000_000, 1_400_000, np.inf],
    labels=[0, 1, 2, 3, 4]).astype(int)

# Trip-risk composite score
df_fe["TripRiskScore"] = (
    df_fe["ChronicDiseases"] * 0.35 +
    df_fe["EverTravelledAbroad"] * 0.25 +
    df_fe["FrequentFlyer"] * 0.20 +
    (df_fe["FamilyMembers"] / 9) * 0.20
)

# Income × Frequent Flyer interaction
df_fe["Income_FF"] = df_fe["AnnualIncome"] * df_fe["FrequentFlyer"] / 1e6

# Age group
df_fe["AgeGroup"] = pd.cut(df_fe["Age"],
    bins=[0, 25, 35, 45, 100], labels=[0, 1, 2, 3]).astype(int)

# High dependents flag
df_fe["HighDependents"] = (df_fe["FamilyMembers"] >= 6).astype(int)

FEATURES = [
    "Age", "EmploymentType", "GraduateOrNot", "AnnualIncome",
    "FamilyMembers", "ChronicDiseases", "FrequentFlyer", "EverTravelledAbroad",
    "IncomeBucket", "TripRiskScore", "Income_FF", "AgeGroup", "HighDependents",
]
TARGET = "TravelInsurance"

X = df_fe[FEATURES].values
y = df_fe[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)

print(f"   Features : {len(FEATURES)}")
print(f"   Train size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")
print(f"   Train positive rate: {y_train.mean():.3f}")

# ─────────────────────────────────────────────────────────────
# 4.  Baseline — Logistic Regression
# ─────────────────────────────────────────────────────────────
print("\n4. Baseline Logistic Regression …")

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, random_state=SEED,
                              class_weight="balanced", C=0.5))
])
lr_pipe.fit(X_train, y_train)
lr_proba = lr_pipe.predict_proba(X_test)[:, 1]

lr_roc   = roc_auc_score(y_test, lr_proba)
lr_prauc = average_precision_score(y_test, lr_proba)
print(f"   LR  ROC-AUC : {lr_roc:.4f}")
print(f"   LR  PR-AUC  : {lr_prauc:.4f}")

# ─────────────────────────────────────────────────────────────
# 5.  XGBoost + GridSearchCV
# ─────────────────────────────────────────────────────────────
print("\n5. XGBoost with hyperparameter tuning …")

scale_pw = (y_train == 0).sum() / (y_train == 1).sum()

xgb_base = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pw,
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
)

param_grid = {
    "n_estimators":    [100, 200],
    "max_depth":       [3, 5],
    "learning_rate":   [0.05, 0.1],
    "subsample":       [0.8, 1.0],
    "colsample_bytree":[0.8, 1.0],
    "min_child_weight":[1, 3],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grid = GridSearchCV(xgb_base, param_grid, cv=cv,
                    scoring="roc_auc", n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)

best_xgb = grid.best_estimator_
print(f"   Best params  : {grid.best_params_}")
print(f"   CV  ROC-AUC  : {grid.best_score_:.4f}")

xgb_proba = best_xgb.predict_proba(X_test)[:, 1]
xgb_roc   = roc_auc_score(y_test, xgb_proba)
xgb_prauc = average_precision_score(y_test, xgb_proba)
print(f"   Test ROC-AUC : {xgb_roc:.4f}")
print(f"   Test PR-AUC  : {xgb_prauc:.4f}")

# ─────────────────────────────────────────────────────────────
# 6.  Calibration
# ─────────────────────────────────────────────────────────────
print("\n6. Probability calibration …")

xgb_calib = CalibratedClassifierCV(
    xgb.XGBClassifier(**grid.best_params_,
                       objective="binary:logistic",
                       scale_pos_weight=scale_pw,
                       random_state=SEED, n_jobs=-1, verbosity=0),
    method="isotonic", cv=5)
xgb_calib.fit(X_train, y_train)
calib_proba = xgb_calib.predict_proba(X_test)[:, 1]

brier_raw   = brier_score_loss(y_test, xgb_proba)
brier_calib = brier_score_loss(y_test, calib_proba)
print(f"   Brier (raw)      : {brier_raw:.4f}")
print(f"   Brier (calibrated): {brier_calib:.4f}")

# ─────────────────────────────────────────────────────────────
# 7.  Threshold Optimisation
# ─────────────────────────────────────────────────────────────
print("\n7. Threshold optimisation …")

thresholds  = np.linspace(0.1, 0.9, 81)
f1_scores   = [f1_score(y_test, (calib_proba >= t).astype(int)) for t in thresholds]
best_thresh = thresholds[np.argmax(f1_scores)]
print(f"   Optimal threshold (max F1) : {best_thresh:.2f}")
print(f"   F1 at optimal              : {max(f1_scores):.4f}")

y_pred_opt = (calib_proba >= best_thresh).astype(int)
print(classification_report(y_test, y_pred_opt, target_names=["No Insurance","Insurance"]))

# ─────────────────────────────────────────────────────────────
# 8.  Evaluation Plots
# ─────────────────────────────────────────────────────────────
print("\n8. Generating evaluation charts …")

# --- 8a. ROC curve ---
fig, ax = plt.subplots(figsize=(7, 6))
for proba, label, color in [
    (lr_proba,    f"Logistic Regression (AUC={lr_roc:.3f})",  AIRBNB_GRAY),
    (xgb_proba,   f"XGBoost raw        (AUC={xgb_roc:.3f})",  AIRBNB_TEAL),
    (calib_proba, f"XGBoost calibrated (AUC={roc_auc_score(y_test, calib_proba):.3f})", AIRBNB_RED),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, lw=2, label=label)

ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.fill_between(*roc_curve(y_test, calib_proba)[:2],
                alpha=0.07, color=AIRBNB_RED)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Insurance Intent Scorer", fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig("charts/03_roc_curve.png", bbox_inches="tight")
plt.close()
print("   Saved charts/03_roc_curve.png")

# --- 8b. PR curve ---
fig, ax = plt.subplots(figsize=(7, 6))
baseline = y_test.mean()
for proba, label, color in [
    (lr_proba,    f"Logistic Regression (AP={lr_prauc:.3f})",  AIRBNB_GRAY),
    (xgb_proba,   f"XGBoost raw        (AP={xgb_prauc:.3f})", AIRBNB_TEAL),
    (calib_proba, f"XGBoost calibrated (AP={average_precision_score(y_test, calib_proba):.3f})", AIRBNB_RED),
]:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax.plot(rec, prec, lw=2, label=label)

ax.axhline(baseline, color="k", ls="--", lw=1, label=f"Random baseline ({baseline:.2f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — Insurance Intent Scorer", fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("charts/04_pr_curve.png", bbox_inches="tight")
plt.close()
print("   Saved charts/04_pr_curve.png")

# --- 8c. Calibration curve ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
n_bins = 10

for proba, label, color in [
    (xgb_proba,   "XGBoost (raw)",        AIRBNB_TEAL),
    (calib_proba, "XGBoost (calibrated)", AIRBNB_RED),
]:
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=n_bins, strategy="quantile")
    axes[0].plot(mean_pred, frac_pos, "s-", lw=2, label=label, color=color)

axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
axes[0].set_xlabel("Mean Predicted Probability")
axes[0].set_ylabel("Fraction of Positives")
axes[0].set_title("Calibration Curve", fontweight="bold")
axes[0].legend()

# F1 vs threshold
axes[1].plot(thresholds, f1_scores, color=AIRBNB_RED, lw=2)
axes[1].axvline(best_thresh, color=AIRBNB_TEAL, ls="--", lw=1.5,
                label=f"Optimal threshold = {best_thresh:.2f}")
axes[1].set_xlabel("Decision Threshold")
axes[1].set_ylabel("F1 Score")
axes[1].set_title("Threshold Optimisation (F1)", fontweight="bold")
axes[1].legend()

plt.suptitle("Calibration & Threshold Analysis", fontsize=13, y=1.01, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/05_calibration_threshold.png", bbox_inches="tight")
plt.close()
print("   Saved charts/05_calibration_threshold.png")

# --- 8d. Confusion matrix ---
cm = confusion_matrix(y_test, y_pred_opt)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["No Insurance", "Insurance"],
            yticklabels=["No Insurance", "Insurance"], ax=ax)
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
ax.set_title(f"Confusion Matrix @ threshold={best_thresh:.2f}", fontweight="bold")
plt.tight_layout()
plt.savefig("charts/06_confusion_matrix.png", bbox_inches="tight")
plt.close()
print("   Saved charts/06_confusion_matrix.png")

# ─────────────────────────────────────────────────────────────
# 9.  SHAP explainability
# ─────────────────────────────────────────────────────────────
print("\n9. SHAP analysis …")

explainer    = shap.TreeExplainer(best_xgb)
shap_values  = explainer.shap_values(X_test)

# --- 9a. SHAP summary (beeswarm) ---
fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(shap_values, X_test, feature_names=FEATURES,
                  show=False, plot_size=None, color_bar=True)
plt.title("SHAP Summary Plot — Feature Impact on Insurance Intent",
          fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("charts/07_shap_summary.png", bbox_inches="tight")
plt.close()
print("   Saved charts/07_shap_summary.png")

# --- 9b. SHAP mean absolute bar ---
mean_shap = np.abs(shap_values).mean(axis=0)
shap_df   = pd.DataFrame({"Feature": FEATURES, "MeanAbsSHAP": mean_shap})
shap_df   = shap_df.sort_values("MeanAbsSHAP", ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(shap_df["Feature"], shap_df["MeanAbsSHAP"], color=AIRBNB_RED)
ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Global Feature Importance (SHAP)", fontweight="bold")
plt.tight_layout()
plt.savefig("charts/08_shap_importance.png", bbox_inches="tight")
plt.close()
print("   Saved charts/08_shap_importance.png")

# --- 9c. SHAP waterfall for highest-propensity guest ---
top_guest_idx = int(np.argmax(calib_proba))
shap_exp = shap.Explanation(
    values          = shap_values[top_guest_idx],
    base_values     = explainer.expected_value,
    data            = X_test[top_guest_idx],
    feature_names   = FEATURES
)
fig, ax = plt.subplots(figsize=(9, 6))
shap.waterfall_plot(shap_exp, max_display=10, show=False)
plt.title(f"SHAP Waterfall — Highest-Intent Guest "
          f"(P={calib_proba[top_guest_idx]:.2f})", fontweight="bold")
plt.tight_layout()
plt.savefig("charts/09_shap_waterfall.png", bbox_inches="tight")
plt.close()
print("   Saved charts/09_shap_waterfall.png")

# ─────────────────────────────────────────────────────────────
# 10.  Decile / Lift chart
# ─────────────────────────────────────────────────────────────
print("\n10. Lift / decile analysis …")

lift_df = pd.DataFrame({"y_true": y_test, "proba": calib_proba})
lift_df["decile"] = pd.qcut(lift_df["proba"].rank(method="first"),
                             q=10, labels=False)
decile_stats = (lift_df.groupby("decile")["y_true"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "UptakeRate", "count": "N"}))
decile_stats["Decile"] = decile_stats["decile"] + 1
decile_stats["Lift"]   = decile_stats["UptakeRate"] / y_test.mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(decile_stats["Decile"], decile_stats["UptakeRate"],
            color=AIRBNB_RED, alpha=0.85)
axes[0].axhline(y_test.mean(), color=AIRBNB_GRAY, ls="--", lw=1.5,
                label=f"Overall rate {y_test.mean():.2f}")
axes[0].set_xlabel("Score Decile (1=lowest risk → 10=highest)")
axes[0].set_ylabel("Insurance Uptake Rate")
axes[0].set_title("Uptake Rate by Score Decile", fontweight="bold")
axes[0].legend()

axes[1].bar(decile_stats["Decile"], decile_stats["Lift"],
            color=AIRBNB_TEAL, alpha=0.85)
axes[1].axhline(1.0, color=AIRBNB_GRAY, ls="--", lw=1.5, label="Baseline (lift=1)")
axes[1].set_xlabel("Score Decile")
axes[1].set_ylabel("Lift")
axes[1].set_title("Lift by Score Decile", fontweight="bold")
axes[1].legend()

plt.suptitle("Decile Analysis — Insurance Intent Scorer", fontsize=13,
             y=1.01, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/10_decile_lift.png", bbox_inches="tight")
plt.close()
print("   Saved charts/10_decile_lift.png")

# ─────────────────────────────────────────────────────────────
# 11.  Save model + metrics
# ─────────────────────────────────────────────────────────────
print("\n11. Saving artefacts …")

joblib.dump(xgb_calib, "insurance_intent_model.pkl")

calib_roc   = roc_auc_score(y_test, calib_proba)
calib_prauc = average_precision_score(y_test, calib_proba)
top10_lift  = decile_stats[decile_stats["Decile"] == 10]["Lift"].values[0]

metrics = {
    "lr_roc_auc":           round(lr_roc, 4),
    "lr_pr_auc":            round(lr_prauc, 4),
    "xgb_raw_roc_auc":      round(xgb_roc, 4),
    "xgb_raw_pr_auc":       round(xgb_prauc, 4),
    "xgb_calib_roc_auc":    round(calib_roc, 4),
    "xgb_calib_pr_auc":     round(calib_prauc, 4),
    "brier_raw":             round(brier_raw, 4),
    "brier_calibrated":      round(brier_calib, 4),
    "optimal_threshold":     round(float(best_thresh), 2),
    "f1_at_optimal":         round(max(f1_scores), 4),
    "top_decile_lift":       round(float(top10_lift), 2),
    "n_features":            len(FEATURES),
    "train_n":               int(X_train.shape[0]),
    "test_n":                int(X_test.shape[0]),
    "positive_rate_test":    round(float(y_test.mean()), 4),
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n" + "="*60)
print("FINAL METRICS SUMMARY")
print("="*60)
for k, v in metrics.items():
    print(f"  {k:30s}: {v}")

print("\nAll charts saved to charts/")
print("Model saved to insurance_intent_model.pkl")
print("Metrics saved to metrics.json")
