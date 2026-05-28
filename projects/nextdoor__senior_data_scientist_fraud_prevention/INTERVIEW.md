# Interview Prep — Senior Data Scientist - Fraud Prevention @ Nextdoor

# Interview Study Sheet — Neighborhood Trust Score: Fraud Detection
### Nextdoor | Senior Data Scientist, Fraud Prevention

This sheet prepares you to discuss the portfolio project honestly, ground answers in real metrics, and connect the work directly to Nextdoor's trust-and-safety mission.

---

## Motivation & Framing

**Q: Why did you build this specific project for this application?**

**A:** Nextdoor's core value proposition is authentic local connection — fake accounts directly corrode that. I wanted to demonstrate that I could build a production-grade fraud classifier end-to-end, not just a notebook experiment, covering imbalanced-class handling, threshold optimization, SHAP explainability, and a reviewer-facing UI. The trust-and-safety framing maps directly to what the fraud prevention team ships: a signal that catches bad actors at registration while giving human reviewers an auditable reason for the flag, not just a black-box score.

**Q: You used a credit card fraud dataset — isn't that a mismatch for account-registration fraud?**

**A:** Completely fair, and I'm upfront about it in the README. The ULB dataset is a structural proxy: it has severe class imbalance (~0.97% fraud), anonymous behavioral signals, and a binary outcome — all analogous to the data shape of registration fraud, even if the underlying domain differs. The honest value of the project isn't "I modeled Nextdoor's data" — it's that I can demonstrate the full methodology, engineering decisions, and operational tooling that I'd apply to Nextdoor's real signals the moment I have access to them.

---

## Technical Choices

**Q: Why XGBoost over alternatives like logistic regression, LightGBM, or a neural network?**

**A:** XGBoost hits a practical sweet spot for fraud detection: it handles the mix of engineered numeric features well, is fast to train and tune (my full pipeline runs in 30–60 seconds including SHAP), and integrates natively with SHAP's TreeExplainer — which gives exact Shapley values rather than approximations. A neural network would sacrifice that interpretability without a meaningful lift at this scale; logistic regression would struggle with the non-linear feature interactions that tree ensembles capture. LightGBM would be a legitimate alternative, and I'd benchmark it against XGBoost on Nextdoor's data, but XGBoost is what I have the deepest production experience with from my Trend Micro work.

**Q: You used both SMOTE and scale_pos_weight — isn't that double-counting your imbalance correction?**

**A:** It's a fair tension to flag. SMOTE synthetically rebalances the training set to 1:1 before model training, so `scale_pos_weight` was tuned down rather than left at the raw 100:1 ratio. In practice I treated them as complementary levers: SMOTE addresses the geometric imbalance in feature space, while `scale_pos_weight` fine-tunes the loss weighting during tree building. I ran ablations to make sure I wasn't over-correcting; the final precision of 0.9560 tells me I didn't push recall so aggressively that I flooded the queue with false alarms.

**Q: Why did you tune the decision threshold rather than just using 0.5, and how did you pick 0.827?**

**A:** With severe class imbalance, 0.5 is almost never the right operating point — it's an artifact of probability calibration, not an operational decision. I swept 200 threshold steps from 0.01 to 0.99 and maximized F1 on the test set, which balances precision and recall symmetrically. The optimizer landed at 0.827, giving me 87 true positives, 4 false positives, and 11 misses out of 98 fraud cases. In production I'd expose this threshold as a tunable parameter — if Nextdoor's review team is understaffed, dial it up for higher precision; if the threat environment spikes, dial it down to catch more fraud at the cost of more analyst time.

**Q: Why SHAP specifically, and what does it add beyond a feature importance bar chart?**

**A:** Standard feature importance tells you which features the model relies on globally, but it doesn't tell a trust-and-safety reviewer *why this account* was flagged. SHAP gives me both: the global beeswarm shows which signals drive fraud across the whole population, and the per-account waterfall shows exactly which features pushed that specific account's risk score above the threshold and by how much. In my Trend Micro pipeline I used SHAP the same way — giving sales leadership an explanation for why a deal was scored high-risk, not just a number. For Nextdoor's reviewers, that auditability matters for policy decisions, appeals handling, and regulatory defensibility.

---

## Results & Honest Limitations

**Q: Walk me through the key metrics — what do they actually mean operationally?**

**A:** On the 10,099-record test set, the model catches 87 of 98 fraud cases (recall 88.8%) with only 4 false alarms (precision 95.6%). That's roughly 1 false alarm per 23 flagged accounts, which is a manageable analyst workload. The ROC-AUC of 0.9818 and PR-AUC of 0.9223 tell me the model has strong discriminative power across the full range of thresholds — PR-AUC is the number I trust most here because it's sensitive to performance on the minority class and doesn't flatter you with true negative volume the way ROC-AUC can in imbalanced settings.

**Q: Where is this model genuinely weak, and what would you improve with more time or real data?**

**A:** Three honest limitations. First, the dataset's features are PCA-anonymized, so I can't engineer domain-specific signals — on Nextdoor's real data I'd build features around registration velocity, device fingerprinting, IP reputation, email domain age, and social graph connectivity, which are far richer than V1–V28. Second, the model is static — there's no concept drift monitoring or scheduled retraining; real fraud patterns evolve and I'd need a drift detector and automated retraining pipeline. Third, I haven't evaluated calibration rigorously: the raw probabilities look reasonable, but I'd run a reliability diagram and add Platt scaling or isotonic regression before using the scores for downstream risk tiering rather than binary flagging.

---

## From Demo to Production

**Q: If Nextdoor gave you access to real registration data tomorrow, what would you do first?**

**A:** The first thing I'd do is a label audit — understanding how Nextdoor currently identifies ground-truth fraud (manual reports, appeals, network analysis?) and how reliable those labels are. Noisy labels kill model quality faster than any algorithmic choice. Second, I'd map the feature space: what signals exist at the moment of registration versus what only becomes available post-hoc? A production model can only use signals available at decision time. Third, I'd run a temporal train/test split rather than random stratification, because random splitting leaks future information and gives you optimistically biased metrics — that's the first thing I'd fix from my current methodology.

**Q: How would you monitor this model once it's live?**

**A:** I'd track three layers. First, data drift on input feature distributions — if registration patterns shift (new device types, new geographies), I want an alert before the model degrades silently. Second, label feedback loops: as trust-and-safety reviewers confirm or overturn flags, those decisions feed back as updated labels for retraining. Third, operational metrics: false positive rate matters to the reviewer team's workload, so I'd build a dashboard showing flagged volume, analyst overturn rate, and fraud rate among passed accounts (estimated from delayed labels). At Trend Micro I built the MLOps quality-gate framework for exactly this kind of continuous assurance — the same pattern applies here.

**Q: How would this scale if Nextdoor's registration volume spikes significantly?**

**A:** XGBoost inference is extremely fast — scoring a single account takes milliseconds, and it's embarrassingly parallelizable. The Streamlit UI in the demo is for analyst review, not the scoring path; in production the model would be served as a REST endpoint behind the registration pipeline. For very high throughput I'd containerize it, put it behind a load balancer, and cache the SHAP explanations for the top flagged accounts rather than computing them synchronously for every request. The retraining pipeline is the harder scaling problem — I'd design it to retrain on a rolling window or triggered by drift alerts rather than on a fixed schedule.

---

## Behavioral & Ownership

**Q: How would you work with Nextdoor's Trust & Safety policy team on something like threshold tuning?**

**A:** I'd frame the threshold conversation explicitly as a business decision, not a modeling decision, and give them the precision-recall tradeoff curve with concrete operational translations: "At 0.827 you're reviewing X accounts per day with Y% confirmed fraud; drop to 0.70 and you catch 5 more fraud cases per week but add Z false positives to the queue." I'd build a simple Streamlit tool — like the one in this project — where the policy lead can simulate the operational impact of different thresholds before we change anything in production. The goal is to make the tradeoff legible to non-technical partners so they own the decision with full information, not to have the data scientist make the call unilaterally.

**Q: Tell me about a time you had to balance model performance against interpretability or operational constraints.**

**A:** At Trend Micro, my executive pipeline intelligence dashboard needed to be trusted by sales leadership, not just accurate. I could have shipped a black-box ensemble with slightly better holdout metrics, but instead I built SHAP explanations into the core output — every deal score came with the top factors driving it up or down. That decision slowed down the initial build by about a week, but it directly drove adoption: leadership used the SHAP-explained swing deals to prioritize $4.4M+ in pipeline, and they wouldn't have trusted a score without understanding why. The same tradeoff applies at Nextdoor — a fraud score that reviewers can't interpret gets ignored or over-ridden, which defeats the purpose of building it.
