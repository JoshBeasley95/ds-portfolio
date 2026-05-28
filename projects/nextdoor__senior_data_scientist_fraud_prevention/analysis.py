"""
Neighborhood Trust Score: Detecting Fake Account Registrations on a Local Social Network
=========================================================================================
Senior Data Scientist – Fraud Prevention | Nextdoor (Portfolio Demo)

Dataset: Credit Card Fraud Detection (ULB, public domain)
         Used as a SYNTHETIC PROXY for account-registration fraud signals.
         This is NOT Nextdoor data.
"""

import os, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, confusion_matrix, f1_score
)
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

import xgboost as xgb
import shap

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
os.makedirs("charts", exist_ok=True)

DARK_BG   = "#0F1117"
PANEL_BG  = "#1A1D27"
ACCENT    = "#00C2FF"
FRAUD_CLR = "#FF4B6E"
LEGIT_CLR = "#00C2FF"
GRID_CLR  = "#2A2D3A"
TEXT_CLR  = "#E8EAF0"

def dark_fig(nrows=1, ncols=1, figsize=(10, 6), **kw):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    fig.patch.set_facecolor(DARK_BG)
    for a in (np.array(ax).flat if hasattr(ax, '__iter__') else [ax]):
        a.set_facecolor(PANEL_BG)
        a.tick_params(colors=TEXT_CLR, labelsize=10)
        for spine in a.spines.values():
            spine.set_edgecolor(GRID_CLR)
        a.xaxis.label.set_color(TEXT_CLR)
        a.yaxis.label.set_color(TEXT_CLR)
        a.title.set_color(TEXT_CLR)
    return fig, ax

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  NEIGHBORHOOD TRUST SCORE — FRAUD DETECTION PIPELINE")
print("=" * 65)
print("\n[1/8] Loading data …")

# Try sklearn's fetch_openml first, then fall back to synthetic generation
try:
    from sklearn.datasets import fetch_openml
    data_bunch = fetch_openml("creditcard", version=1, as_frame=True, parser="auto")
    df_raw = data_bunch.frame.copy()
    df_raw.columns = [c.lower() for c in df_raw.columns]
    df_raw["class"] = df_raw["class"].astype(int)
    print(f"    Loaded from OpenML: {df_raw.shape}")
    DATA_SOURCE = "OpenML Credit Card Fraud (ULB, public domain)"
except Exception as e:
    print(f"    OpenML unavailable ({e}). Generating synthetic proxy dataset …")
    # ── Synthetic proxy ────────────────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    n_legit, n_fraud = 28000, 492
    n_total = n_legit + n_fraud

    # 28 PCA-like features (V1–V28) + Time + Amount
    V_legit = rng.standard_normal((n_legit, 28))
    V_fraud = rng.standard_normal((n_fraud, 28)) + rng.uniform(-3, 3, (1, 28))
    V = np.vstack([V_legit, V_fraud])

    time_vals = np.concatenate([
        rng.uniform(0, 172800, n_legit),
        rng.choice(np.concatenate([rng.uniform(0, 50000, n_fraud//2),
                                   rng.uniform(120000, 172800, n_fraud//2)]), n_fraud)
    ])
    amount_legit = np.abs(rng.lognormal(3.5, 1.4, n_legit))
    amount_fraud = np.abs(rng.lognormal(4.2, 1.8, n_fraud))
    amounts = np.concatenate([amount_legit, amount_fraud])
    labels  = np.array([0]*n_legit + [1]*n_fraud)

    cols = [f"v{i}" for i in range(1, 29)] + ["time", "amount", "class"]
    df_raw = pd.DataFrame(np.column_stack([V, time_vals, amounts, labels]), columns=cols)
    df_raw["class"] = df_raw["class"].astype(int)
    print(f"    Synthetic proxy created: {df_raw.shape}")
    DATA_SOURCE = "Synthetic proxy (mirrors ULB creditcard structure)"

# ── Subsample to keep runtime fast ────────────────────────────────────────────
fraud_df = df_raw[df_raw["class"] == 1]
legit_df = df_raw[df_raw["class"] == 0].sample(min(50000, len(df_raw[df_raw["class"]==0])), random_state=SEED)
df = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"    Working dataset: {df.shape}  |  fraud={df['class'].sum()} ({df['class'].mean()*100:.2f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  EDA — CLASS IMBALANCE & FEATURE DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/8] EDA & class-imbalance visualisation …")

fraud_count = df["class"].value_counts()
fig, axes = dark_fig(1, 2, figsize=(13, 5))

# 2a — Bar chart
bar_colors = [LEGIT_CLR, FRAUD_CLR]
bars = axes[0].bar(["Legitimate (0)", "Fraud (1)"], fraud_count.values,
                   color=bar_colors, width=0.5, edgecolor=PANEL_BG, linewidth=1.5)
axes[0].set_title("Class Distribution (Account Events)", fontsize=13, fontweight="bold", pad=12)
axes[0].set_ylabel("Count")
axes[0].yaxis.grid(True, color=GRID_CLR, linestyle="--", alpha=0.5)
axes[0].set_axisbelow(True)
for bar, v in zip(bars, fraud_count.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 200,
                 f"{v:,}\n({v/len(df)*100:.1f}%)",
                 ha="center", va="bottom", color=TEXT_CLR, fontsize=10, fontweight="bold")

# 2b — Amount distributions by class
legit_amt = np.log1p(df[df["class"]==0]["amount"])
fraud_amt = np.log1p(df[df["class"]==1]["amount"])
axes[1].hist(legit_amt, bins=60, alpha=0.7, color=LEGIT_CLR, label="Legitimate", density=True)
axes[1].hist(fraud_amt, bins=60, alpha=0.7, color=FRAUD_CLR, label="Fraud",      density=True)
axes[1].set_title("log(Amount+1) Distribution by Class", fontsize=13, fontweight="bold", pad=12)
axes[1].set_xlabel("log(Amount + 1)")
axes[1].set_ylabel("Density")
axes[1].yaxis.grid(True, color=GRID_CLR, linestyle="--", alpha=0.5)
axes[1].set_axisbelow(True)
leg = axes[1].legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=10)

fig.suptitle("Neighborhood Trust Score — Data Overview", fontsize=14, fontweight="bold",
             color=TEXT_CLR, y=1.02)
plt.tight_layout()
fig.savefig("charts/class_imbalance.png", dpi=150, bbox_inches="tight",
            facecolor=DARK_BG, edgecolor="none")
plt.close()
print("    ✓  charts/class_imbalance.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/8] Feature engineering …")

df["log_amount"] = np.log1p(df["amount"])

# Time-based features: use 'time' column if present, else create a synthetic proxy
if "time" in df.columns:
    df["hour_of_day"] = (df["time"].astype(float) % 86400) / 3600
else:
    # Synthetic time proxy: uniform random hours (time column absent in this OpenML version)
    rng_t = np.random.default_rng(SEED)
    df["hour_of_day"] = rng_t.uniform(0, 24, size=len(df))

df["sin_hour"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
df["cos_hour"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

# Amount velocity proxies (binned quantile buckets)
df["amount_bucket"] = pd.qcut(df["amount"].astype(float), q=10, labels=False, duplicates="drop")

# Amount z-score (how unusual is this amount relative to the population?)
amt_f = df["amount"].astype(float)
df["amount_zscore"] = (amt_f - amt_f.mean()) / (amt_f.std() + 1e-9)

# Drop raw columns subsumed by engineered features
v_cols = [c for c in df.columns if c.startswith("v")]
feature_cols = v_cols + ["log_amount", "sin_hour", "cos_hour", "amount_bucket", "amount_zscore"]

X = df[feature_cols].astype(float)
y = df["class"].astype(int)

print(f"    Feature matrix: {X.shape}  |  target positives: {y.sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAIN / TEST SPLIT + SMOTE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/8] Train/test split + SMOTE oversampling …")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

smote = SMOTE(random_state=SEED, k_neighbors=5)
X_tr_sm, y_tr_sm = smote.fit_resample(X_train, y_train)
print(f"    Before SMOTE  — train: {X_train.shape}  fraud={y_train.sum()}")
print(f"    After  SMOTE  — train: {X_tr_sm.shape}  fraud={y_tr_sm.sum()}")
print(f"    Test set      — {X_test.shape}  fraud={y_test.sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  XGBOOST TRAINING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/8] Training XGBoost classifier …")

scale_pos = (y_tr_sm == 0).sum() / (y_tr_sm == 1).sum()  # ~1 after SMOTE, but kept for clarity

model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    use_label_encoder=False,
    eval_metric="aucpr",
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
)

t0 = time.time()
model.fit(
    X_tr_sm, y_tr_sm,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
elapsed = time.time() - t0
print(f"    Training complete in {elapsed:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  THRESHOLD OPTIMISATION + EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/8] Evaluation & threshold optimisation …")

y_proba = model.predict_proba(X_test)[:, 1]

# Sweep thresholds for best F1
thresholds = np.linspace(0.01, 0.99, 200)
f1_scores  = [f1_score(y_test, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
best_idx   = int(np.argmax(f1_scores))
best_thresh = thresholds[best_idx]
best_f1    = f1_scores[best_idx]

y_pred = (y_proba >= best_thresh).astype(int)

roc_auc  = roc_auc_score(y_test, y_proba)
avg_prec = average_precision_score(y_test, y_proba)
cm       = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
precision_at_thresh = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_at_thresh    = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\n    ── Key Metrics ──────────────────────────────────")
print(f"    ROC-AUC              : {roc_auc:.4f}")
print(f"    PR-AUC (Avg Prec)    : {avg_prec:.4f}")
print(f"    Best F1 @ threshold  : {best_f1:.4f}  (threshold={best_thresh:.3f})")
print(f"    Precision            : {precision_at_thresh:.4f}")
print(f"    Recall               : {recall_at_thresh:.4f}")
print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")

report = classification_report(y_test, y_pred, target_names=["Legitimate","Fraud"])
print(f"\n{report}")

# ── Plot ROC + PR curves ───────────────────────────────────────────────────────
fpr_vals, tpr_vals, _ = roc_curve(y_test, y_proba)
prec_vals, rec_vals, pr_thresh = precision_recall_curve(y_test, y_proba)

fig, axes = dark_fig(1, 2, figsize=(14, 6))

# ROC
axes[0].plot(fpr_vals, tpr_vals, color=ACCENT, lw=2.5, label=f"XGBoost  AUC={roc_auc:.4f}")
axes[0].plot([0,1],[0,1], color=GRID_CLR, lw=1.5, linestyle="--", label="Random baseline")
axes[0].set_xlabel("False Positive Rate");  axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve — Account Fraud Detection", fontsize=12, fontweight="bold")
axes[0].legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
axes[0].yaxis.grid(True, color=GRID_CLR, linestyle="--", alpha=0.4)
axes[0].xaxis.grid(True, color=GRID_CLR, linestyle="--", alpha=0.4)
axes[0].set_axisbelow(True)

# PR
axes[1].plot(rec_vals, prec_vals, color=FRAUD_CLR, lw=2.5, label=f"XGBoost  AUCPR={avg_prec:.4f}")
axes[1].axhline(y=y_test.mean(), color=GRID_CLR, lw=1.5, linestyle="--", label="Random baseline")
# Mark best-F1 threshold
axes[1].scatter([recall_at_thresh], [precision_at_thresh],
                color="#FFD700", s=120, zorder=5, label=f"Best-F1 threshold ({best_thresh:.2f})")
axes[1].set_xlabel("Recall");  axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve — Account Fraud Detection", fontsize=12, fontweight="bold")
axes[1].legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)
axes[1].yaxis.grid(True, color=GRID_CLR, linestyle="--", alpha=0.4)
axes[1].xaxis.grid(True, color=GRID_CLR, linestyle="--", alpha=0.4)
axes[1].set_axisbelow(True)

fig.suptitle("Model Performance — Neighborhood Trust Score", fontsize=13, fontweight="bold",
             color=TEXT_CLR)
plt.tight_layout()
fig.savefig("charts/pr_roc_curves.png", dpi=150, bbox_inches="tight",
            facecolor=DARK_BG, edgecolor="none")
plt.close()
print("\n    ✓  charts/pr_roc_curves.png")

# ── Confusion matrix ──────────────────────────────────────────────────────────
fig2, ax2 = dark_fig(1, 1, figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred Legit","Pred Fraud"],
            yticklabels=["True Legit","True Fraud"],
            ax=ax2, cbar=False, linewidths=0.5, linecolor=GRID_CLR,
            annot_kws={"size": 14, "color": "white", "fontweight": "bold"})
ax2.set_title(f"Confusion Matrix  (threshold={best_thresh:.2f})", fontsize=12,
              fontweight="bold", color=TEXT_CLR)
ax2.tick_params(colors=TEXT_CLR)
plt.tight_layout()
fig2.savefig("charts/confusion_matrix.png", dpi=150, bbox_inches="tight",
             facecolor=DARK_BG, edgecolor="none")
plt.close()
print("    ✓  charts/confusion_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/8] SHAP explainability …")

# Use a sample for speed
shap_sample_size = min(2000, len(X_test))
X_shap = X_test.iloc[:shap_sample_size].reset_index(drop=True)

explainer   = shap.TreeExplainer(model)
shap_values = explainer(X_shap)   # returns Explanation object

# ── 7a  SHAP Summary (Beeswarm) ────────────────────────────────────────────────
print("    Computing SHAP summary plot …")
fig3, ax3 = plt.subplots(figsize=(11, 8))
fig3.patch.set_facecolor(DARK_BG)
ax3.set_facecolor(PANEL_BG)

shap.summary_plot(
    shap_values.values,
    X_shap,
    feature_names=feature_cols,
    plot_type="dot",
    max_display=15,
    show=False,
    color_bar=True,
)
ax3 = plt.gca()
ax3.set_facecolor(PANEL_BG)
fig3.patch.set_facecolor(DARK_BG)
plt.title("SHAP Feature Importance — Top 15 Fraud Drivers\n(Neighborhood Trust Score)",
          color=TEXT_CLR, fontsize=12, fontweight="bold", pad=10)
plt.tick_params(colors=TEXT_CLR)
ax3.xaxis.label.set_color(TEXT_CLR)
ax3.yaxis.label.set_color(TEXT_CLR)
for spine in ax3.spines.values():
    spine.set_edgecolor(GRID_CLR)

plt.tight_layout()
plt.savefig("charts/shap_summary.png", dpi=150, bbox_inches="tight",
            facecolor=DARK_BG, edgecolor="none")
plt.close()
print("    ✓  charts/shap_summary.png")

# ── 7b  SHAP Waterfall — single high-risk account ─────────────────────────────
print("    Computing SHAP waterfall for highest-risk account …")

# Find the highest-fraud-probability record in shap sample
proba_shap = model.predict_proba(X_shap)[:, 1]
top_fraud_idx = int(np.argmax(proba_shap))

fig4, ax4 = plt.subplots(figsize=(11, 7))
fig4.patch.set_facecolor(DARK_BG)
ax4.set_facecolor(PANEL_BG)

shap.waterfall_plot(shap_values[top_fraud_idx], max_display=12, show=False)

ax4 = plt.gca()
ax4.set_facecolor(PANEL_BG)
fig4 = plt.gcf()
fig4.patch.set_facecolor(DARK_BG)
plt.tick_params(colors=TEXT_CLR)
ax4.xaxis.label.set_color(TEXT_CLR)
ax4.yaxis.label.set_color(TEXT_CLR)
for spine in ax4.spines.values():
    spine.set_edgecolor(GRID_CLR)

prob_val = proba_shap[top_fraud_idx]
plt.title(f"SHAP Waterfall — Highest-Risk Account\n(Fraud Probability = {prob_val:.3f})",
          color=TEXT_CLR, fontsize=12, fontweight="bold", pad=10)

plt.tight_layout()
plt.savefig("charts/shap_waterfall_example.png", dpi=150, bbox_inches="tight",
            facecolor=DARK_BG, edgecolor="none")
plt.close()
print("    ✓  charts/shap_waterfall_example.png")

# ── 7c  Feature importance (gain) bar chart ────────────────────────────────────
mean_abs_shap = pd.Series(
    np.abs(shap_values.values).mean(axis=0),
    index=feature_cols
).sort_values(ascending=True).tail(15)

fig5, ax5 = dark_fig(1, 1, figsize=(10, 7))
bars = ax5.barh(mean_abs_shap.index, mean_abs_shap.values,
                color=ACCENT, edgecolor=PANEL_BG, height=0.7)
ax5.set_xlabel("Mean |SHAP value|", color=TEXT_CLR)
ax5.set_title("Top 15 Features by Mean |SHAP| Value\n(Fraud Signal Strength)",
              color=TEXT_CLR, fontsize=12, fontweight="bold")
ax5.xaxis.grid(True, color=GRID_CLR, linestyle="--", alpha=0.4)
ax5.set_axisbelow(True)
for bar, v in zip(bars, mean_abs_shap.values):
    ax5.text(v + 0.0005, bar.get_y() + bar.get_height()/2,
             f"{v:.4f}", va="center", color=TEXT_CLR, fontsize=9)
plt.tight_layout()
fig5.savefig("charts/shap_importance_bar.png", dpi=150, bbox_inches="tight",
             facecolor=DARK_BG, edgecolor="none")
plt.close()
print("    ✓  charts/shap_importance_bar.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8.  SAVE METRICS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8/8] Saving metrics summary …")

metrics_dict = {
    "roc_auc":              float(roc_auc),
    "pr_auc":               float(avg_prec),
    "best_f1":              float(best_f1),
    "best_threshold":       float(best_thresh),
    "precision_at_thresh":  float(precision_at_thresh),
    "recall_at_thresh":     float(recall_at_thresh),
    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    "n_train":              int(len(X_train)),
    "n_test":               int(len(X_test)),
    "fraud_rate_test":      float(y_test.mean()),
    "top_shap_feature":     str(mean_abs_shap.index[-1]),
    "data_source":          DATA_SOURCE,
}

import json
with open("metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)
print("    ✓  metrics.json")

print("\n" + "="*65)
print("  PIPELINE COMPLETE")
print(f"  ROC-AUC  = {roc_auc:.4f}")
print(f"  PR-AUC   = {avg_prec:.4f}")
print(f"  Best F1  = {best_f1:.4f}  @  threshold {best_thresh:.3f}")
print(f"  Precision = {precision_at_thresh:.4f} | Recall = {recall_at_thresh:.4f}")
print("="*65)
