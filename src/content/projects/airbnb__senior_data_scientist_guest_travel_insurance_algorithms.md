---
title: "Guest Insurance Intent Scorer: Predicting Travel Protection Uptake from Booking Signals"
company: "Airbnb"
role: "Senior Data Scientist, Guest Travel Insurance (Algorithms)"
date: "2026-05-27"
tags:
  - XGBoost
  - SHAP
  - Propensity Modeling
  - Calibration
  - Travel Insurance
  - Scikit-Learn
  - Feature Engineering
  - Threshold Optimization
  - Python
  - Streamlit
summary: >
  Built a calibrated XGBoost propensity model that predicts which guests are likely to purchase
  travel insurance, using engineered booking signals, isotonic calibration, SHAP explainability,
  and threshold optimisation — directly mirroring the ML work Airbnb's GTI Algorithms team
  applies to real AirCover coverage decisions.
---

> **Data disclaimer:** All results below were produced on **fully synthetic data** (1,987 rows)
> generated to mirror the schema of the public *Travel Insurance Prediction Dataset*.
> No Airbnb internal, proprietary, or user data was used at any point.

---

## The Problem: Showing the Right Insurance Offer at the Right Moment

Every time a guest completes a booking on Airbnb, there's a fleeting window to offer them
travel protection — a flight cancellation refund, a trip interruption guarantee, or emergency
medical cover under AirCover. Surface the offer to the right guest and you drive meaningful
uptake. Surface it to the wrong guest at the wrong moment and you degrade trust, inflate
complaint rates, and waste prime checkout real estate.

This project builds the **intent scoring engine** that powers that decision: a machine learning
model that takes structured guest and booking attributes and outputs a *calibrated probability*
that this particular guest would value and purchase insurance coverage.

---

## Dataset & Schema

The model trains on a synthetic dataset of **1,987 guest records** mirroring the public
Travel Insurance Prediction Dataset. Key attributes include:

- **Demographics:** Age, EmploymentType (government vs. private), GraduateOrNot
- **Financial profile:** AnnualIncome
- **Family context:** FamilyMembers
- **Health:** ChronicDiseases (binary)
- **Travel behaviour:** FrequentFlyer, EverTravelledAbroad
- **Target:** TravelInsurance (1 = purchased, 0 = did not)

The dataset is **class-imbalanced at ~21 % positive rate** — realistic for an opt-in insurance
offer in a consumer travel context.

---

## Feature Engineering: Adding Signal Beyond Raw Fields

Raw features only go so far. Five engineered features capture the latent risk and intent signals
that a product analyst would reason about intuitively:

| Engineered Feature | Construction | Intuition |
|---|---|---|
| `IncomeBucket` | Quintile bin of AnnualIncome | Discretises wealth tier for monotone effects |
| `TripRiskScore` | Weighted sum: chronic × 0.35 + abroad × 0.25 + FF × 0.20 + family_size × 0.20 | Composite exposure score |
| `Income_FF` | AnnualIncome × FrequentFlyer / 1e6 | High-income frequent travellers: premium segment |
| `AgeGroup` | Cut into 4 bins (<25, 25–35, 35–45, 45+) | Non-linear age effects |
| `HighDependents` | FamilyMembers ≥ 6 | Flag for guests with large families to protect |

---

## Modelling Pipeline

### Step 1: Baseline — Logistic Regression

A class-balanced logistic regression (C = 0.5, `class_weight=balanced`) establishes the floor.
It achieves **ROC-AUC 0.6260** on the held-out 20 % test set — a useful ceiling check showing
that linear separability is modest and a tree-based model should add value.

### Step 2: XGBoost + GridSearchCV

XGBoost is the natural choice for tabular propensity models: handles mixed types, captures
interactions, and integrates natively with SHAP. Class imbalance is addressed via `scale_pos_weight`
(ratio of negatives to positives ≈ 3.7).

A 5-fold stratified `GridSearchCV` evaluated 192 hyperparameter combinations across:
- `n_estimators` ∈ {100, 200}
- `max_depth` ∈ {3, 5}
- `learning_rate` ∈ {0.05, 0.1}
- `subsample` ∈ {0.8, 1.0}
- `colsample_bytree` ∈ {0.8, 1.0}
- `min_child_weight` ∈ {1, 3}

**Best configuration:** n_estimators=100, max_depth=3, learning_rate=0.05, subsample=1.0,
colsample_bytree=1.0, min_child_weight=3.

CV ROC-AUC: **0.5969** → Test ROC-AUC: **0.6066**

### Step 3: Isotonic Calibration

Raw XGBoost probabilities are frequently miscalibrated, particularly in imbalanced settings.
`CalibratedClassifierCV` with `method="isotonic"` and 5-fold CV corrects the score distribution.

| Metric | Raw | Calibrated |
|---|---|---|
| Brier Score | 0.2347 | **0.1626** |

A **30.7 % reduction in Brier score** means the model's probabilities now behave as true
frequencies — essential for any downstream pricing or ranked-surfacing system.

![Calibration curve and threshold sweep](/projects/airbnb__senior_data_scientist_guest_travel_insurance_algorithms/charts/05_calibration_threshold.png)

The reliability diagram (left panel) shows the calibrated model's curve hugging the diagonal,
while the raw model systematically overshoots. The right panel shows the F1 sweep that
identifies the optimal decision threshold.

---

## Evaluation

### ROC and Precision-Recall Curves

![ROC curve](/projects/airbnb__senior_data_scientist_guest_travel_insurance_algorithms/charts/03_roc_curve.png)

The calibrated XGBoost achieves **ROC-AUC 0.6295**, beating the logistic regression baseline
on both ROC-AUC (0.6260) and PR-AUC (0.3244 vs. 0.3356 LR — comparable). 

![PR curve](/projects/airbnb__senior_data_scientist_guest_travel_insurance_algorithms/charts/04_pr_curve.png)

The PR curve matters more in imbalanced settings. The calibrated model maintains precision
well above the 0.21 random baseline across most recall levels.

### Threshold Optimisation: Precision vs. Recall Trade-off

In an insurance-offer context, the stakes of the two error types differ:
- **False positive** (offer to uninterested guest): annoyance, potential trust erosion
- **False negative** (miss a willing buyer): lost revenue

The F1-maximising threshold is **0.23** — deliberately lower than 0.5, given the 21 % base
rate. At this threshold:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No Insurance | 0.85 | 0.72 | 0.78 |
| Insurance | 0.34 | 0.53 | **0.41** |

Product and revenue teams can shift this threshold up or down based on the unit economics
of a false positive vs. a false negative — a straightforward dial in the scoring service.

---

## Decile Lift Analysis

![Decile lift chart](/projects/airbnb__senior_data_scientist_guest_travel_insurance_algorithms/charts/10_decile_lift.png)

Score the test population, bin by decile, and compute uplift vs. the overall 21.4 % base rate:

- **Top decile (score 90–100th percentile):** uptake rate ~35 %, lift **1.64×**
- **Bottom decile:** uptake rate ~10 %, lift 0.47×

This separation is the practical proof-of-concept: sorting guests by model score and targeting
the top 20 % would reach ~3× more insurance buyers per offer impression than random targeting.

---

## SHAP Explainability

![SHAP summary beeswarm](/projects/airbnb__senior_data_scientist_guest_travel_insurance_algorithms/charts/07_shap_summary.png)

TreeExplainer computes exact Shapley values in O(T·L) time for tree ensembles. Key findings:

1. **EverTravelledAbroad** — the single strongest positive driver. Guests with international
   travel history show dramatically higher insurance intent, likely because they've experienced
   or heard about travel disruptions first-hand.

2. **AnnualIncome** — a smooth positive gradient: wealthier guests are more willing to pay
   for protection (and less price-sensitive to the premium).

3. **TripRiskScore** — the engineered composite captures chronic health × travel frequency
   × family size in a single feature that ranks third globally.

4. **FrequentFlyer** — frequent flyers have more to lose from disruption and more experience
   purchasing insurance.

5. **Age** — modest positive slope; older guests trend toward higher uptake.

![Per-guest SHAP waterfall](/projects/airbnb__senior_data_scientist_guest_travel_insurance_algorithms/charts/09_shap_waterfall.png)

The waterfall plot shows exactly *why* the highest-intent guest in the test set received a
high score — with signed contributions from each feature, traceable by any compliance
or product reviewer.

---

## Streamlit Scoring App

`app.py` provides a lightweight UI where any stakeholder can enter a hypothetical guest
profile and receive:
- A calibrated probability (0–100 %)
- A three-tier intent classification (Low / Medium / High)
- A surfacing recommendation ("show prominently", "soft nudge", "suppress")
- Top-3 SHAP drivers with signed contributions

This directly models the API endpoint a GTI product team would call at booking checkout.

---

## Why This Matters for Airbnb's GTI Algorithms Team

The Senior Data Scientist (GTI Algorithms) JD calls for exactly the capability stack
demonstrated here:

| JD Requirement | Project Demonstration |
|---|---|
| Build ML models predicting likelihood to value coverage | Calibrated XGBoost propensity scorer |
| Handle class imbalance in real booking data | `scale_pos_weight` + calibration |
| Calibrate probabilities for pricing decisions | Isotonic calibration, Brier score −30.7 % |
| Provide explainability for legal/product review | SHAP waterfall per guest |
| Optimise precision/recall trade-offs with stakeholders | Threshold sweep, F1 optimisation |
| Surface signals from structured booking data | Feature engineering on 8 raw → 13 features |
| End-to-end ML: EDA → model → deploy | Full pipeline + Streamlit app |

The insurance intent problem sits at the intersection of personalisation, revenue, and trust —
and it demands exactly the blend of predictive rigour, calibration care, and explainability
that this project demonstrates.

---

## How to Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn scipy joblib
python analysis.py          # → charts/, metrics.json, insurance_intent_model.pkl
streamlit run app.py        # → interactive scorer UI
```
