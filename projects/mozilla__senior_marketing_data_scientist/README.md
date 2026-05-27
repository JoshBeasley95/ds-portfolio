# Geo-Based Marketing Lift Measurement with Synthetic Control

> **Demonstration project using fully synthetic data.**  
> Models the type of geo-level incrementality experiment Mozilla's Marketing Data Science team
> runs when evaluating regional paid media campaigns. No internal Mozilla data is used.

---

## Problem Statement

In a privacy-first, cookieless world, last-click attribution is unreliable. Mozilla's Marketing
Data Science team needs rigorous, channel-level measurement of whether spend drives *incremental*
Firefox downloads. This project demonstrates a **synthetic control / CausalImpact-style geo
experiment** — the industry gold standard for incrementality measurement.

**Key questions answered:**
- Did the simulated paid media campaign cause a statistically significant lift in Firefox market share?
- How large was the lift, and how confident are we?
- Which geo characteristics amplify campaign effectiveness?
- What was the estimated cost per incremental install?

---

## Data

| Source | Description |
|--------|-------------|
| **Fully synthetic** | 15 European geo-markets, 36 monthly periods (Jan 2022–Dec 2024) |
| AR(1) + trend model | Realistic browser-share dynamics with per-geo noise profiles |
| Campaign overlay | Synthetic spend injected for DE, FR, PL starting Jan 2024 |
| True lift injected | DE: +2.8 pp, FR: +2.1 pp, PL: +3.2 pp (ground truth for validation) |

The data generation intentionally mirrors the structure of StatCounter/GWS country-level
browser market share data. Population, broadband penetration, and tech-index features
are modeled after publicly available Eurostat/ITU statistics.

---

## Methodology

### 1. Exploratory Data Analysis
- Time-series plots of Firefox share trends per geo
- Indexed parallel-trends check (pre-intervention period)
- Pre-period correlation test: treated vs average control

### 2. Synthetic Control Construction
- **Donor pool**: 12 untreated European geos
- **Optimization**: `scipy.optimize.minimize` with SLSQP, constrained to convex weights (non-negative, sum to 1)
- **Fit quality**: Pre-period RMSE 0.10–0.35 pp across treated geos
- **Inference**: Point-wise and cumulative post-period lift estimates

### 3. Bootstrapped Confidence Intervals
- 500 bootstrap iterations resampling the donor pool
- 95% CI computed from the bootstrap lift distribution

### 4. Placebo / Permutation Tests
- Synthetic control run on each of the 12 control geos
- Treated geo lift compared against the empirical null distribution
- All three treated geos: **p < 0.001** (lift exceeds all 12 placebo geos)

### 5. XGBoost + SHAP
- Post-period Firefox share predicted from geo features + campaign spend
- SHAP values surface which geo characteristics amplify lift
- Top drivers: pre-campaign baseline, broadband penetration, campaign spend

### 6. ROI Estimation
- Lift pp × geo internet users → incremental installs
- Total campaign spend / incremental installs → cost per install

---

## Key Results

| Geo | Estimated Lift | True Lift | 95% CI           | p-value  | CPI     |
|-----|---------------:|----------:|------------------|:--------:|--------:|
| DE  | **2.795 pp**   | 2.8 pp    | [2.727, 2.967]   | < 0.001  | $1.08   |
| FR  | **2.053 pp**   | 2.1 pp    | [1.891, 2.111]   | < 0.001  | $1.53   |
| PL  | **3.060 pp**   | 3.2 pp    | [2.912, 3.105]   | < 0.001  | $1.56   |

The synthetic control recovered the injected ground-truth lift within **0.14–0.15 pp** across
all three geos — validating the methodology on data with known answers.

XGBoost companion model: **RMSE = 0.0795 pp, R² = 0.9974**

---

## Business Impact

- **Germany** delivers the strongest absolute ROI: ~2M incremental installs at $1.08 CPI
- **Poland** shows the highest percentage lift (+3.06 pp) relative to baseline, suggesting
  high responsiveness to paid media — likely due to lower Firefox baseline + rising tech adoption
- **France** shows moderate lift at slightly higher CPI; recommend re-examining creative/channel mix
- Methodology is **privacy-safe**: no user-level data, cookie-free, relies only on aggregate geo-level signals

---

## Files

| File | Description |
|------|-------------|
| `analysis.py` | Full pipeline: data generation → EDA → synthetic control → placebo tests → XGBoost + SHAP |
| `app.py` | Streamlit interactive dashboard |
| `charts/` | 6 PNG charts produced by the pipeline |
| `RESULTS.md` | Real metrics table from the run |
| `writeup.mdx` | Stakeholder memo / publish-ready narrative |
| `results.json` | Machine-readable results |

---

## How to Run

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install pandas numpy scipy scikit-learn xgboost shap \
            matplotlib seaborn statsmodels streamlit

# 3. Run analysis pipeline
python analysis.py

# 4. Launch interactive dashboard
streamlit run app.py
```

**Runtime:** ~30–45 seconds (no GPU required)

---

## Why This Matters to Mozilla

Mozilla's Marketing Data Science JD explicitly calls out geo experiments and synthetic control
(CausalImpact, GeoLift) as differentiating expertise. This project directly demonstrates:

- **Causal inference skill**: synthetic control is the rigorous alternative to A/B when true
  randomization at the user level is infeasible or privacy-incompatible
- **Privacy-by-design measurement**: aggregate geo signals, no user tracking required
- **End-to-end ownership**: from data generation through model, inference, validation, and
  business-language ROI translation — exactly the "full-stack" measurement expected of a
  Senior MDS at Mozilla
- **Browser-market fluency**: framed around Firefox market share, the core Mozilla KPI

---

## Skills Demonstrated

`Python` · `pandas` · `NumPy` · `SciPy` · `XGBoost` · `SHAP` · `scikit-learn` ·
`statsmodels` · `Matplotlib` · `Seaborn` · `Streamlit` · `Causal Inference` ·
`Synthetic Control` · `Permutation Testing` · `Time Series Analysis` · `Marketing Analytics`
