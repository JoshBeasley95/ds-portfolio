# Interview Prep — Senior Data Scientist, Guest Travel Insurance (Algorithms) @ Airbnb

# Guest Insurance Intent Scorer — Interview Study Sheet

This sheet prepares you to discuss the portfolio project confidently and honestly in a Senior Data Scientist, Guest Travel Insurance (Algorithms) interview at Airbnb. Every metric cited is real; every limitation is acknowledged directly.

---

## Motivation & Framing

**Q: Why did you build this specific project for this role?**

**A:** The GTI Algorithms job description is explicit about needing calibrated propensity models that predict whether a guest will value coverage, handle class imbalance, and produce explainable outputs for legal and product partners. I wanted to demonstrate exactly that stack end-to-end — not just fit a model, but show calibration, threshold optimisation, SHAP explainability, and decile lift in one cohesive pipeline. It's a direct map to the role's stated technical requirements rather than a generic classification demo.

**Q: The dataset is synthetic and small — why should I take the methodology seriously?**

**A:** I'm fully transparent that this uses ~1,987 rows of synthetic data mirroring the public Travel Insurance Prediction Dataset schema, and no Airbnb data was used at any point. The value isn't the specific numbers — it's the end-to-end design: the calibration step, the threshold sweep logic, the SHAP layer, and the decile-lift framing are all directly portable to real booking-signal data. Think of it as a blueprint that I'd scale and retrain the moment I had access to Airbnb's actual features and labels.

---

## Technical Choices

**Q: Why XGBoost over a simpler model or a neural approach?**

**A:** With under 2,000 rows and nine raw features, a neural network would overfit badly and provide no interpretability advantage. XGBoost with modest depth (max_depth=3, selected by GridSearchCV over 192 combinations) gives strong regularisation, handles mixed feature types without heavy preprocessing, and integrates cleanly with TreeExplainer for exact SHAP values — which I needed for the per-guest explainability requirement. Logistic regression served as my baseline and actually had a comparable ROC-AUC of 0.626, which is an honest signal that the dataset is noisy and small, not that XGBoost is dramatically superior here.

**Q: How did you handle class imbalance, and why that approach?**

**A:** The positive rate was 21.4%, so moderately imbalanced. I used `scale_pos_weight` in XGBoost to up-weight the minority class during training, and I also set `class_weight='balanced'` on the logistic regression baseline. Importantly, I evaluated on PR-AUC alongside ROC-AUC, because PR-AUC is more sensitive to minority-class performance — a model that just predicts "no insurance" everywhere would look fine on ROC but terrible on precision-recall. The calibrated model's PR-AUC came in at 0.3244 versus a random baseline of roughly 0.21, which is meaningful but not dramatic given the dataset limitations.

**Q: Why did you add isotonic calibration, and did it actually matter?**

**A:** Calibration matters enormously for an insurance use case because the raw probability feeds downstream decisions — whether to surface an offer, what price to show, how to rank guests in a queue. Raw XGBoost probabilities are notoriously overconfident. Isotonic calibration via `CalibratedClassifierCV` reduced the Brier score from 0.2347 to 0.1626, a 30.7% improvement, which means the model's confidence estimates are substantially closer to true observed rates. That's not cosmetic — if you're pricing a personalised offer based on a 70% score that's actually a 50% probability, you're mispricing the product.

**Q: Walk me through the threshold optimisation decision.**

**A:** At a 21.4% base rate, the default 0.5 threshold is almost useless — it would classify almost everyone as "no purchase." I swept the threshold from 0.1 to 0.9 and maximised F1, landing on 0.23 as optimal, which gives F1 = 0.41, precision = 0.34, and recall = 0.53. The low threshold reflects the low base rate and a deliberate choice to favour recall: I'd rather surface an offer to some guests who don't buy than miss the high-intent guests entirely, as long as precision is high enough not to annoy users. In a real Airbnb setting, I'd frame that tradeoff explicitly with the product team — the right threshold depends on the revenue per conversion versus the cost of a spurious offer.

**Q: Why SHAP specifically, and what did it tell you here?**

**A:** SHAP gives me additive, model-consistent feature attributions rather than the impurity-based importance that XGBoost reports natively, which can be misleading for correlated features. I used TreeExplainer for exact computation. Globally, `EverTravelledAbroad` and `AnnualIncome` are the two strongest drivers — guests who've travelled internationally and earn more show much higher insurance intent. I also built per-guest waterfall plots so that any individual offer decision can be explained in plain language for a compliance or product review: "This guest was scored high primarily because they've travelled abroad before and have a high income." That's directly relevant to Airbnb's need for legally reviewable offer decisions.

---

## Results & Honest Limitations

**Q: The ROC-AUC is 0.63 — that's not very high. How do you defend it?**

**A:** I don't spin it. A 0.63 ROC-AUC on synthetic data with eight raw features and under 2,000 rows is modest, and I'm honest about that in the project. The more meaningful number for this problem is the decile lift: top-decile guests are 1.64× more likely to purchase than average, which is a real, actionable signal for targeting. In production on Airbnb's data — with hundreds of behavioural signals, booking history, browse patterns, and cancellation history — I'd expect substantially stronger discrimination. The point of the demo is the architecture, not claiming state-of-the-art performance on a toy dataset.

**Q: What are the biggest limitations you'd call out proactively?**

**A:** Three honest ones. First, the dataset is tiny and synthetic, so the model's learned patterns may not generalise to anything real. Second, I have no temporal validation — in a real deployment, I'd need a time-based train/test split to prevent leakage from future booking behaviour and to test whether the model degrades over time. Third, the feature set is generic demographic data; the real signal at Airbnb would come from behavioural features — trip value, cancellation rate, destination risk, trip length — that I simply don't have here.

---

## From Demo to Production

**Q: If Airbnb gave you access to real booking data tomorrow, what would you change first?**

**A:** The first thing I'd do is audit the label definition — "purchased insurance" is clean in this dataset, but in real data I'd want to understand whether the label captures offer acceptance, actual policy bind, or something else, and whether there's selection bias from which guests even saw the offer. Then I'd rebuild the feature set around booking signals: trip value, lead time, destination country risk tier, guest's prior cancellation history, and whether the guest has travelled with companions. Those features are far more predictive than demographics alone, and they're directly observable from Airbnb's systems at the moment of offer decision.

**Q: How would you handle monitoring and model drift in production?**

**A:** I'd set up three layers of monitoring. First, input drift monitoring on feature distributions — if `EverTravelledAbroad` suddenly shifts because of a product change or a global travel shock, I want an alert before the model silently degrades. Second, calibration monitoring on rolling windows: check whether the model's predicted probabilities match observed uptake rates weekly. Third, outcome monitoring on the business metric — offer acceptance rate by score decile. At Trend Micro I built an MLOps framework with automated quality gates and eval rubrics for exactly this kind of continuous validation, and I'd apply the same structure here.

**Q: The GTI role involves working with legal and compliance partners. How does this project prepare you for that?**

**A:** The SHAP waterfall plots are the direct answer — they produce a per-guest, human-readable explanation of why a specific score was generated, which is the kind of artifact a compliance team can review when a guest asks "why was I shown this offer?" or "why was my claim processed this way?" I've also thought about the threshold as a policy lever: the precision/recall tradeoff isn't just a model parameter, it's a fairness and user-experience decision that should be made jointly with legal and product. At Trend Micro I regularly translated model outputs into executive narratives and sales tools, so I'm practiced at bridging the gap between what the model says and what a non-technical partner needs to act on it.

---

## Behavioral & Ownership

**Q: Tell me about a time you owned a model end-to-end and drove real adoption.**

**A:** At Trend Micro I built an executive pipeline intelligence dashboard powered by two XGBoost models trained on 60,000+ deals with SHAP explainability, blended ML conversion probabilities with historical rates, and added gap-to-plan decomposition. The output wasn't just a Jupyter notebook — it was an interactive dashboard that US sales leadership adopted to identify $4.4M+ in swing deals and drive their weekly forecasting calls. The lesson I carry forward is that a model only generates impact when the stakeholders trust it and know how to act on it, which means investing in calibration, explainability, and the right presentation layer — exactly the same priorities this insurance project demonstrates.

**Q: What tradeoff did you make in this project that you'd revisit with more time?**

**A:** I'd revisit the calibration method choice. I used isotonic calibration because it's flexible and worked well here, but isotonic can overfit on small datasets — with ~400 test samples it's a real concern. With more time I'd compare Platt scaling and temperature scaling, run the calibration on a held-out fold rather than the full test set, and report reliability diagrams at multiple bin widths to make sure the improvement isn't an artifact of the bin size. I made the pragmatic call given the dataset size, but I'd want to validate that more rigorously before deploying something that affects pricing decisions.
