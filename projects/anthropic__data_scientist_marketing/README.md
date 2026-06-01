# Lifecycle Segment Lift: Causal Experiment Design for AI Product Onboarding

A demonstration of how a marketing data scientist would design an A/B experiment,
build behavioral user segments, and measure whether a lifecycle intervention
(an onboarding email) actually caused conversion change, not just correlated with it.

Built on synthetic data that mirrors the SaaS/AI product lifecycle problem at companies
like Anthropic, where the team needs rigorous causal thinking, not just dashboards.

---

## The Problem

Onboarding emails go to whoever the CRM targets. That group tends to be more engaged
to start with. So when you look at the raw conversion gap between email-recipients and
non-recipients, you're measuring a mix of two things: the email's real effect plus the
pre-existing advantage of the users who got it. Propensity score matching separates them.

This project works through the full stack:

1. Generating realistic synthetic data with built-in confounding
2. Segmenting users by behavior before any experiment analysis
3. Estimating a propensity score for who received the email
4. Matching treated and control users on that score to get an unbiased ATE
5. Training an XGBoost classifier on the matched sample and explaining it with SHAP
6. Surfacing segment-level lift estimates in a Streamlit app

---

## Key Results

All numbers below are from a real pipeline run, not estimates.

**Causal Lift (Propensity Score Matching)**

| Method | ATE |
|---|---|
| Naive (raw difference) | +0.159 |
| PSM-Matched (causal) | +0.077 |

The naive estimate overstates the email's effect by about 2x. After matching, the
true lift is roughly 7.7 percentage points, still meaningful but a lot more honest.
Covariate imbalance on sessions dropped from 2.44 to 0.12 post-match.

**Conversion Prediction (XGBoost, 5-fold CV on matched sample)**

- AUROC: 0.798
- AUPRC: 0.713

**Top conversion drivers (SHAP):**

Sessions in the last 14 days dominates everything else (mean |SHAP| = 0.805).
Pro plan adds strong lift (0.326). Days since signup and messages sent are close
in importance (0.253, 0.246). The email itself ranks 5th (0.199), which is
consistent with it having a real but moderate causal effect.

**Segment Lift:**

| Segment | Predicted Email Lift |
|---|---|
| Power Users | +0.092 |
| Engaged Tenured | +0.067 |
| Moderate Users | +0.066 |
| Low-Activity New | +0.059 |

Power Users get the most lift from the email, not because they convert at higher
baseline rates (though they do at 63.6%), but because they have more room to
respond to the treatment in the model's learned relationship.

---

## Approach

### Data

Synthetic, 5,000 users, fixed seed (42). Features: days since signup, sessions in
last 14 days, messages sent, plan type, industry. Confounding is baked in: the email
treatment was assigned via a logistic model that favors more active users, mimicking
real CRM targeting behavior.

### Segmentation

KMeans (k=4) on sessions, messages, and tenure. Four natural clusters emerged:
Low-Activity New users (highest volume, lowest conversion), Moderate Users, Engaged
Tenured (high message volume), and Power Users (high sessions, by far the best converters).

### Causal Estimation

1. Fit a logistic regression to predict P(email | user features)
2. Match each treated user to its nearest-neighbor control by propensity score
3. Compute the ATE on the matched sample

This removes the selection bias from CRM targeting. The matched ATE (0.077) is the
honest number to report to stakeholders.

### Modeling + Explainability

XGBoost trained on the matched sample. SHAP TreeExplainer shows sessions as the
clear top driver, consistent with the data generating process. The email feature
places 5th, which is also consistent: it matters, but behavioral engagement signals
carry more predictive weight.

---

## Scope Note

This is a demonstration on synthetic data only. The data was generated in code with
a fixed random seed. It is not real user data from any company. The problem structure
(SaaS lifecycle, onboarding emails, propensity-matched experiments) mirrors real
marketing DS work at AI product companies.

---

## How to Run

```bash
# 1. Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn streamlit scipy

# 3. Run the full analysis pipeline
python analysis.py
# -> produces charts/ and results.json

# 4. Launch the Streamlit app
streamlit run app.py
```

---

## File Structure

```
.
├── generate_data.py          # Reproducible synthetic dataset
├── analysis.py               # Full pipeline: EDA, segmentation, PSM, model, SHAP
├── app.py                    # Streamlit cohort explorer
├── charts/
│   ├── eda_conversion_by_segment.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_feature_distributions.png
│   ├── segmentation_elbow.png
│   ├── segmentation_heatmap.png
│   ├── causal_ate_comparison.png
│   ├── propensity_score_overlap.png
│   ├── model_roc_pr_curves.png
│   ├── shap_beeswarm.png
│   ├── shap_importance_bar.png
│   ├── shap_dependence.png
│   └── lift_by_segment.png
├── RESULTS.md
└── README.md
```

---

## Why This Matters for Anthropic's Marketing DS Role

The role asks for experiment design, causal inference studies, and user segmentation
to measure marketing intervention effectiveness. This project covers that exact stack:

- Propensity matching rather than raw A/B comparisons, because real targeting is
  never random
- Segment-level lift estimates, because the "average user" is usually not your target
- SHAP explainability, so the model's output is useful to non-technical stakeholders
- A Streamlit app, so a growth PM can actually use the analysis without touching code

The problem is framed around AI product onboarding, the exact context of Claude's
user lifecycle, where understanding whether an email actually changed behavior
(versus just correlating with users who would have converted anyway) is the
difference between good and bad marketing spend decisions.
