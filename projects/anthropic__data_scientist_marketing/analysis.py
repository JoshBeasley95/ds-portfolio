"""
analysis.py
Full pipeline: EDA -> Segmentation -> Propensity Matching -> XGBoost -> SHAP
All charts saved to charts/. Prints real metrics used in RESULTS.md.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, classification_report
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency

import xgboost as xgb
import shap

from generate_data import generate_dataset

# ── Setup ──────────────────────────────────────────────────────────────────
CHARTS = Path("charts")
CHARTS.mkdir(exist_ok=True)

DARK_BG   = "#0E1117"
ACCENT    = "#E87C3E"
ACCENT2   = "#4C9ED9"
GRID_CLR  = "#2A2D3A"
TEXT_CLR  = "#FAFAFA"

def dark_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor":   DARK_BG,
        "axes.edgecolor":   GRID_CLR,
        "axes.labelcolor":  TEXT_CLR,
        "xtick.color":      TEXT_CLR,
        "ytick.color":      TEXT_CLR,
        "text.color":       TEXT_CLR,
        "grid.color":       GRID_CLR,
        "grid.linewidth":   0.5,
        "axes.grid":        True,
        "legend.facecolor": "#1A1D27",
        "legend.edgecolor": GRID_CLR,
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.labelsize":   11,
    })

dark_style()

RESULTS = {}

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)
df = generate_dataset()
print(f"Shape: {df.shape}")
print(f"Conversion rate : {df['converted_to_paid'].mean():.3f}")
print(f"Email treat rate: {df['received_onboarding_email'].mean():.3f}")

RESULTS["n_users"] = len(df)
RESULTS["overall_conversion_rate"] = round(df["converted_to_paid"].mean(), 4)
RESULTS["email_treatment_rate"] = round(df["received_onboarding_email"].mean(), 4)

# ══════════════════════════════════════════════════════════════════════════════
# 2. EDA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. EDA")
print("=" * 60)

# 2a. Conversion rate by segment (plan x industry)
conv_plan = df.groupby("plan_type")["converted_to_paid"].mean().rename("conversion_rate")
conv_ind  = df.groupby("industry")["converted_to_paid"].mean().rename("conversion_rate")
print("\nConversion by plan:\n", conv_plan.to_string())
print("\nConversion by industry:\n", conv_ind.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(DARK_BG)

colors_plan = [ACCENT if v == conv_plan.max() else ACCENT2 for v in conv_plan.values]
axes[0].bar(conv_plan.index, conv_plan.values, color=colors_plan, edgecolor=GRID_CLR, linewidth=0.8)
axes[0].set_title("Conversion Rate by Plan Type")
axes[0].set_ylabel("Conversion Rate")
axes[0].set_ylim(0, 1)
for i, v in enumerate(conv_plan.values):
    axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", color=TEXT_CLR, fontsize=10)

colors_ind = [ACCENT if v == conv_ind.max() else ACCENT2 for v in conv_ind.values]
axes[1].bar(conv_ind.index, conv_ind.values, color=colors_ind, edgecolor=GRID_CLR, linewidth=0.8)
axes[1].set_title("Conversion Rate by Industry")
axes[1].set_ylabel("Conversion Rate")
axes[1].set_ylim(0, 1)
for i, v in enumerate(conv_ind.values):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", color=TEXT_CLR, fontsize=10)

plt.tight_layout()
plt.savefig(CHARTS / "eda_conversion_by_segment.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("Saved: eda_conversion_by_segment.png")

# 2b. Correlation heatmap
num_cols = ["days_since_signup", "sessions_last_14d", "messages_sent",
            "received_onboarding_email", "converted_to_paid"]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(DARK_BG)
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, ax=ax, linewidths=0.5, linecolor=GRID_CLR,
    annot_kws={"size": 10},
    cbar_kws={"shrink": 0.8}
)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(CHARTS / "eda_correlation_heatmap.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("Saved: eda_correlation_heatmap.png")

# 2c. Feature distributions by conversion status
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.patch.set_facecolor(DARK_BG)
feat_names = ["sessions_last_14d", "messages_sent", "days_since_signup"]
for ax, feat in zip(axes, feat_names):
    for val, color, label in [(0, ACCENT2, "Not Converted"), (1, ACCENT, "Converted")]:
        subset = df[df["converted_to_paid"] == val][feat]
        ax.hist(subset, bins=30, alpha=0.6, color=color, label=label, density=True)
    ax.set_title(feat.replace("_", " ").title())
    ax.set_xlabel(feat)
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(CHARTS / "eda_feature_distributions.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("Saved: eda_feature_distributions.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. BEHAVIORAL SEGMENTATION (KMeans)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. BEHAVIORAL SEGMENTATION")
print("=" * 60)

seg_feats = ["sessions_last_14d", "messages_sent", "days_since_signup"]
scaler = StandardScaler()
X_seg = scaler.fit_transform(df[seg_feats])

# Elbow: k=2..8
inertias = []
ks = range(2, 9)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_seg)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor(DARK_BG)
ax.plot(list(ks), inertias, marker="o", color=ACCENT, linewidth=2, markersize=8)
ax.axvline(4, color=ACCENT2, linestyle="--", alpha=0.7, label="chosen k=4")
ax.set_title("KMeans Elbow Curve")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia")
ax.legend()
plt.tight_layout()
plt.savefig(CHARTS / "segmentation_elbow.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("Saved: segmentation_elbow.png")

# Final clustering with k=4
K = 4
km4 = KMeans(n_clusters=K, random_state=42, n_init=10)
df["segment"] = km4.fit_predict(X_seg)

# Segment profiles
seg_profile = df.groupby("segment").agg(
    n=("user_id", "count"),
    avg_sessions=("sessions_last_14d", "mean"),
    avg_messages=("messages_sent", "mean"),
    avg_tenure=("days_since_signup", "mean"),
    conversion_rate=("converted_to_paid", "mean"),
    email_rate=("received_onboarding_email", "mean"),
).round(3)

# Sort by conversion rate and assign meaningful labels
seg_profile = seg_profile.sort_values("conversion_rate")
label_map = {
    seg_profile.index[0]: "Low-Activity New",
    seg_profile.index[1]: "Moderate Users",
    seg_profile.index[2]: "Engaged Tenured",
    seg_profile.index[3]: "Power Users",
}
df["segment_label"] = df["segment"].map(label_map)
seg_profile["label"] = seg_profile.index.map(label_map)
print("\nSegment Profiles:")
print(seg_profile.to_string())

RESULTS["segments"] = seg_profile[["label", "n", "avg_sessions", "avg_messages",
                                     "avg_tenure", "conversion_rate"]].to_dict("index")

# Segment heatmap
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(DARK_BG)
heatmap_data = seg_profile[["avg_sessions", "avg_messages", "avg_tenure",
                              "conversion_rate", "email_rate"]].copy()
heatmap_data.index = seg_profile["label"]
heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
sns.heatmap(
    heatmap_norm.T, annot=heatmap_data.T, fmt=".2f", cmap="YlOrRd",
    ax=ax, linewidths=0.5, linecolor=GRID_CLR,
    annot_kws={"size": 10},
)
ax.set_title("User Segment Profiles (normalized, annotated with raw values)")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(CHARTS / "segmentation_heatmap.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("Saved: segmentation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PROPENSITY SCORE MATCHING + CAUSAL LIFT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. PROPENSITY SCORE MATCHING")
print("=" * 60)

# Encode categoricals
df_ps = df.copy()
df_ps["plan_pro"] = (df_ps["plan_type"] == "pro").astype(int)
df_ps["ind_tech"] = (df_ps["industry"] == "tech").astype(int)
df_ps["ind_fin"]  = (df_ps["industry"] == "finance").astype(int)
df_ps["ind_hlth"] = (df_ps["industry"] == "healthcare").astype(int)

ps_features = ["sessions_last_14d", "messages_sent", "days_since_signup",
               "plan_pro", "ind_tech", "ind_fin", "ind_hlth"]

X_ps = df_ps[ps_features].values
T = df_ps["received_onboarding_email"].values.astype(int)
Y = df_ps["converted_to_paid"].values.astype(int)

# Fit propensity model
ps_model = LogisticRegression(max_iter=500, random_state=42)
ps_scaler = StandardScaler()
X_ps_scaled = ps_scaler.fit_transform(X_ps)
ps_model.fit(X_ps_scaled, T)
df_ps["propensity_score"] = ps_model.predict_proba(X_ps_scaled)[:, 1]

print(f"Propensity model C-stat: {roc_auc_score(T, df_ps['propensity_score']):.3f}")
RESULTS["propensity_auroc"] = round(roc_auc_score(T, df_ps["propensity_score"]), 4)

# Naive ATE (raw difference in means)
naive_ate = (
    df_ps[df_ps["received_onboarding_email"] == 1]["converted_to_paid"].mean()
    - df_ps[df_ps["received_onboarding_email"] == 0]["converted_to_paid"].mean()
)
print(f"\nNaive ATE (biased): {naive_ate:+.4f}")

# Propensity score matching: nearest-neighbor 1:1 without replacement
treated   = df_ps[df_ps["received_onboarding_email"] == 1].copy()
control   = df_ps[df_ps["received_onboarding_email"] == 0].copy()

from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn.fit(control[["propensity_score"]].values)
distances, indices = nn.kneighbors(treated[["propensity_score"]].values)

matched_control_idx = control.index[indices.flatten()]
matched_treated  = treated.copy()
matched_control  = control.loc[matched_control_idx].copy()

matched_ate = (
    matched_treated["converted_to_paid"].mean()
    - matched_control["converted_to_paid"].mean()
)
print(f"Matched ATE (debiased): {matched_ate:+.4f}")
print(f"Treated n={len(matched_treated)}, Control n={len(matched_control)}")

# Check balance post-matching
balance_before = abs(
    treated["sessions_last_14d"].mean() - control["sessions_last_14d"].mean()
)
balance_after = abs(
    matched_treated["sessions_last_14d"].mean() - matched_control["sessions_last_14d"].mean()
)
print(f"\nSessions imbalance before: {balance_before:.3f}")
print(f"Sessions imbalance after : {balance_after:.3f}")

RESULTS["naive_ate"] = round(float(naive_ate), 4)
RESULTS["matched_ate"] = round(float(matched_ate), 4)
RESULTS["propensity_n_treated"] = int(len(matched_treated))
RESULTS["propensity_n_control"] = int(len(matched_control))

# ATE comparison bar chart
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(DARK_BG)
labels = ["Naive ATE\n(unadjusted)", "PSM-Matched ATE\n(causal estimate)"]
values = [naive_ate, matched_ate]
colors = [ACCENT, ACCENT2]
bars = ax.bar(labels, values, color=colors, width=0.45, edgecolor=GRID_CLR, linewidth=0.8)
ax.axhline(0, color=TEXT_CLR, linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_title("Naive vs. Causal ATE: Onboarding Email → Conversion")
ax.set_ylabel("Average Treatment Effect (Δ conversion rate)")
for bar, val in zip(bars, values):
    ypos = val + 0.002 if val > 0 else val - 0.006
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f"{val:+.4f}", ha="center", fontsize=12, color=TEXT_CLR, fontweight="bold")
plt.tight_layout()
plt.savefig(CHARTS / "causal_ate_comparison.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("Saved: causal_ate_comparison.png")

# Propensity score distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor(DARK_BG)
for ax, title, t_df, c_df in [
    (axes[0], "Before Matching", treated, control),
    (axes[1], "After Matching",  matched_treated, matched_control),
]:
    ax.hist(t_df["propensity_score"], bins=30, alpha=0.6, color=ACCENT,
            label="Treated", density=True)
    ax.hist(c_df["propensity_score"], bins=30, alpha=0.6, color=ACCENT2,
            label="Control", density=True)
    ax.set_title(f"Propensity Score Distribution: {title}")
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Density")
    ax.legend()
plt.tight_layout()
plt.savefig(CHARTS / "propensity_score_overlap.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("Saved: propensity_score_overlap.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. XGBOOST CONVERSION CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. XGBOOST MODEL + EVALUATION")
print("=" * 60)

# Build matched sample for modeling
matched_df = pd.concat([matched_treated, matched_control], ignore_index=True)

model_features = ["sessions_last_14d", "messages_sent", "days_since_signup",
                  "plan_pro", "ind_tech", "ind_fin", "ind_hlth",
                  "received_onboarding_email"]

X_model = matched_df[model_features].values
y_model = matched_df["converted_to_paid"].values.astype(int)

# Cross-validated predictions for unbiased metrics
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False,
    verbosity=0,
)

y_prob_cv = cross_val_predict(clf, X_model, y_model, cv=cv, method="predict_proba")[:, 1]
y_pred_cv = (y_prob_cv >= 0.5).astype(int)

auroc = roc_auc_score(y_model, y_prob_cv)
auprc = average_precision_score(y_model, y_prob_cv)
print(f"CV AUROC:  {auroc:.4f}")
print(f"CV AUPRC:  {auprc:.4f}")
print(classification_report(y_model, y_pred_cv))

RESULTS["xgb_cv_auroc"] = round(float(auroc), 4)
RESULTS["xgb_cv_auprc"] = round(float(auprc), 4)

# Fit final model on full matched sample for SHAP
clf.fit(X_model, y_model)

# ROC + PR curves
fpr, tpr, _ = roc_curve(y_model, y_prob_cv)
prec, rec, _ = precision_recall_curve(y_model, y_prob_cv)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(DARK_BG)

axes[0].plot(fpr, tpr, color=ACCENT, linewidth=2.5, label=f"XGBoost (AUROC={auroc:.3f})")
axes[0].plot([0, 1], [0, 1], color=GRID_CLR, linestyle="--", linewidth=1.5, label="Random")
axes[0].set_title("ROC Curve")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

axes[1].plot(rec, prec, color=ACCENT2, linewidth=2.5, label=f"XGBoost (AUPRC={auprc:.3f})")
axes[1].axhline(y_model.mean(), color=GRID_CLR, linestyle="--", linewidth=1.5, label="Baseline")
axes[1].set_title("Precision-Recall Curve")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()

plt.tight_layout()
plt.savefig(CHARTS / "model_roc_pr_curves.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("Saved: model_roc_pr_curves.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. SHAP EXPLAINABILITY")
print("=" * 60)

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_model)

feat_labels = ["Sessions (14d)", "Messages Sent", "Days Since Signup",
               "Plan: Pro", "Industry: Tech", "Industry: Finance",
               "Industry: Healthcare", "Email Received"]

# SHAP beeswarm
shap_df = pd.DataFrame(shap_values, columns=feat_labels)
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
print("Feature importance (mean |SHAP|):")
for feat, val in mean_abs_shap.items():
    print(f"  {feat:<25} {val:.4f}")

RESULTS["shap_top_features"] = mean_abs_shap.head(5).round(4).to_dict()

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)

# Manual beeswarm-style dot plot
order = mean_abs_shap.index.tolist()
for i, feat in enumerate(reversed(order)):
    vals = shap_df[feat].values
    feat_vals = X_model[:, feat_labels.index(feat)]
    # normalize feature values for color
    fv_norm = (feat_vals - feat_vals.min()) / (np.ptp(feat_vals) + 1e-9)
    jitter = np.random.default_rng(0).uniform(-0.3, 0.3, size=len(vals))
    scatter = ax.scatter(vals, i + jitter, c=fv_norm, cmap="coolwarm",
                         alpha=0.4, s=8, vmin=0, vmax=1)

ax.set_yticks(range(len(order)))
ax.set_yticklabels(list(reversed(order)), fontsize=10)
ax.axvline(0, color=TEXT_CLR, linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xlabel("SHAP Value (impact on conversion log-odds)")
ax.set_title("SHAP Beeswarm: Feature Impact on Paid Conversion")
cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Feature Value (low → high)", color=TEXT_CLR)
cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
plt.tight_layout()
plt.savefig(CHARTS / "shap_beeswarm.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("Saved: shap_beeswarm.png")

# SHAP mean bar
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(DARK_BG)
sorted_feats = mean_abs_shap.sort_values()
colors = [ACCENT if i >= len(sorted_feats) - 3 else ACCENT2
          for i in range(len(sorted_feats))]
ax.barh(sorted_feats.index, sorted_feats.values, color=colors,
        edgecolor=GRID_CLR, linewidth=0.6)
ax.set_title("Mean |SHAP| Feature Importance")
ax.set_xlabel("Mean |SHAP Value|")
plt.tight_layout()
plt.savefig(CHARTS / "shap_importance_bar.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("Saved: shap_importance_bar.png")

# SHAP dependence: Sessions vs conversion
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(DARK_BG)
for ax, feat_idx, feat_name in [
    (axes[0], 0, "Sessions (14d)"),
    (axes[1], 1, "Messages Sent"),
]:
    x_vals = X_model[:, feat_idx]
    s_vals = shap_values[:, feat_idx]
    sc = ax.scatter(x_vals, s_vals, c=y_model, cmap="coolwarm",
                    alpha=0.4, s=10)
    ax.set_xlabel(feat_name)
    ax.set_ylabel("SHAP Value")
    ax.set_title(f"SHAP Dependence: {feat_name}")
    ax.axhline(0, color=TEXT_CLR, linewidth=0.8, linestyle="--", alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Converted (1) / Not (0)")
plt.tight_layout()
plt.savefig(CHARTS / "shap_dependence.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("Saved: shap_dependence.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. SEGMENT-LEVEL LIFT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. SEGMENT-LEVEL LIFT")
print("=" * 60)

# Compute predicted lift = P(convert | email=1) - P(convert | email=0) per segment
df_full = df_ps.copy()
df_full["plan_pro"] = (df_full["plan_type"] == "pro").astype(int)
df_full["ind_tech"]  = (df_full["industry"] == "tech").astype(int)
df_full["ind_fin"]   = (df_full["industry"] == "finance").astype(int)
df_full["ind_hlth"]  = (df_full["industry"] == "healthcare").astype(int)
df_full["segment_label"] = df["segment_label"]

X_treat1 = df_full[model_features].copy()
X_treat0 = df_full[model_features].copy()
X_treat1["received_onboarding_email"] = 1
X_treat0["received_onboarding_email"] = 0

p1 = clf.predict_proba(X_treat1.values)[:, 1]
p0 = clf.predict_proba(X_treat0.values)[:, 1]
df_full["predicted_lift"] = p1 - p0

lift_by_seg = df_full.groupby("segment_label")["predicted_lift"].mean().sort_values(ascending=False)
print("\nPredicted lift by segment:")
print(lift_by_seg.round(4).to_string())

RESULTS["lift_by_segment"] = lift_by_seg.round(4).to_dict()

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(DARK_BG)
colors_seg = [ACCENT if v == lift_by_seg.max() else ACCENT2 for v in lift_by_seg.values]
bars = ax.bar(lift_by_seg.index, lift_by_seg.values, color=colors_seg,
              edgecolor=GRID_CLR, linewidth=0.8)
ax.set_title("Predicted Email Lift by User Segment")
ax.set_ylabel("Avg Predicted ΔP(Conversion)")
ax.set_xlabel("User Segment")
for bar, val in zip(bars, lift_by_seg.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
            f"{val:.3f}", ha="center", color=TEXT_CLR, fontsize=10)
plt.tight_layout()
plt.savefig(CHARTS / "lift_by_segment.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("Saved: lift_by_segment.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. SAVE RESULTS JSON
# ══════════════════════════════════════════════════════════════════════════════
with open("results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print("\n" + "=" * 60)
print("PIPELINE COMPLETE — RESULTS SUMMARY")
print("=" * 60)
print(f"  Users:            {RESULTS['n_users']:,}")
print(f"  Conversion rate:  {RESULTS['overall_conversion_rate']:.1%}")
print(f"  Naive ATE:        {RESULTS['naive_ate']:+.4f}")
print(f"  Matched ATE:      {RESULTS['matched_ate']:+.4f}")
print(f"  Propensity AUROC: {RESULTS['propensity_auroc']:.4f}")
print(f"  XGB CV AUROC:     {RESULTS['xgb_cv_auroc']:.4f}")
print(f"  XGB CV AUPRC:     {RESULTS['xgb_cv_auprc']:.4f}")
print(f"\nTop SHAP features:")
for feat, val in RESULTS["shap_top_features"].items():
    print(f"  {feat:<25} {val:.4f}")

print("\nCharts saved:")
for p in sorted(CHARTS.glob("*.png")):
    print(f"  {p}")
