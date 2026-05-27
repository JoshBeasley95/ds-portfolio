# Interview Prep — Senior Marketing Data Scientist @ Mozilla

# Interview Prep: Geo-Based Marketing Lift Measurement with Synthetic Control

This sheet prepares you to discuss the project confidently and honestly — grounded in real outputs, candid about synthetic-data limitations, and always connecting back to what you'd do differently on Mozilla's actual data.

---

## Motivation & Framing

**Q: Why did you build this project, and why does it map to this role?**

**A:** Mozilla's Marketing Data Science JD specifically calls out geo experiments and tools like CausalImpact and GeoLift as differentiating skills. I wanted to demonstrate that I understand not just the tooling but the *why* — in a privacy-first, cookieless world, user-level attribution is increasingly unreliable, and aggregate geo-level incrementality measurement is the rigorous alternative. The project frames everything around Firefox market share, which is Mozilla's core growth KPI, so it's as close to the real business question as I could get on public/synthetic data.

**Q: Why is synthetic control the right frame for Firefox download measurement specifically?**

**A:** True randomized A/B tests require you to randomly assign treatment at some unit level — but you can't randomly assign a country to see a campaign, and you wouldn't want to because of spillover and market-coherence concerns. Synthetic control constructs a data-driven counterfactual from a donor pool of untreated geos, which is exactly how incrementality gets measured when randomization is infeasible. It's also privacy-safe by design: everything operates on aggregate geo-level signals, no user tracking or cookies required — which aligns directly with Mozilla's values around user privacy.

---

## Technical Choices

**Q: How does synthetic control actually work, and why did you choose it over a simpler diff-in-diff?**

**A:** Synthetic control uses constrained optimization — I used `scipy.optimize.minimize` with SLSQP — to find a convex combination of donor-pool geos (non-negative weights summing to 1) that best reproduces the treated geo's pre-intervention trend. Diff-in-diff assumes a single parallel-trends intercept and treats all controls equally; synthetic control is more flexible because it lets the pre-period data itself determine which controls matter and how much, which produces a tighter counterfactual. My pre-period RMSE ranged from 0.10 pp for France to 0.35 pp for Poland, which is tight enough that the post-period divergence is clearly attributable to the campaign, not pre-existing trend differences.

**Q: How did you validate that the methodology actually works?**

**A:** Because I generated the data synthetically, I had ground truth — I injected known lifts of 2.8 pp for Germany, 2.1 pp for France, and 3.2 pp for Poland. The synthetic control recovered those lifts within 0.14–0.15 pp across all three geos (estimates of 2.795, 2.053, and 3.060 pp respectively), which is strong methodological validation. On top of that, I ran permutation/placebo tests on all 12 control geos: their post-period "lifts" under synthetic control were near-zero, and all three treated geos exceeded the entire empirical null distribution — giving p < 0.001 for each. Bootstrapped 95% CIs from 500 donor-pool resamples further quantified uncertainty around each point estimate.

**Q: Why did you add an XGBoost + SHAP layer on top of the causal inference piece?**

**A:** The synthetic control answers "did the campaign work and how much?" The XGBoost companion model answers a different question: "which geo characteristics predict post-campaign performance, and what levers matter for geo selection in future campaigns?" It's essentially a planning tool — the top SHAP drivers were pre-campaign baseline share, broadband penetration, and campaign spend, which makes intuitive sense and gives the media planning team actionable inputs for deciding where to spend next. The model fit was extremely tight (RMSE 0.0795 pp, R² = 0.9974), though I want to be transparent that high R² is partly expected given the synthetic data structure — on real data with messier dynamics, I'd expect that to soften.

**Q: How did you translate lift estimates into business-language ROI?**

**A:** Lift in percentage points × the geo's internet user population gives an estimate of incremental installs; dividing total campaign spend by that gives cost per install. Germany came out at $1.08 CPI on roughly 2 million incremental installs, France at $1.53, and Poland at $1.56. Poland actually showed the highest percentage lift at 3.06 pp — I attributed that in the write-up to a lower Firefox baseline combined with rising tech adoption, suggesting high responsiveness to paid media. These are the numbers a CMO or Head of Growth actually cares about, so anchoring the output in CPI and incremental volume was a deliberate stakeholder-communication choice.

---

## Results & Honest Limitations

**Q: What are the real limitations of this project, and where would the methodology be weakest on live data?**

**A:** The biggest limitation is that the data is fully synthetic and generated with an AR(1) + trend model I designed myself — so the ground-truth recovery looks clean partly because the data-generating process is well-behaved. Real browser market share data from StatCounter or GWS has noisier cross-geo correlations, potential structural breaks (browser updates, news events), and donor pools that may not span the full pre-period cleanly. The 12-geo donor pool is also small by synthetic-control standards; with real data I'd want 20–30 geos and a longer pre-period to stabilize the weights. Poland's pre-period RMSE of 0.35 pp was notably higher than the others, which is a signal that the synthetic control fit was looser there — in production I'd flag that as requiring closer scrutiny before trusting the lift estimate.

**Q: What would you improve if you had more time or real data?**

**A:** A few things immediately: I'd implement Jackknife or time-series leave-one-out cross-validation on the pre-period to get a more honest read on counterfactual stability, and I'd add a formal RMSPE ratio test (the Abadie/Diamond/Hainmueller approach) as a complement to the permutation test. I'd also explore Bayesian structural time-series (the CausalImpact library) as a parallel methodology — having two independent causal estimates agree gives much stronger confidence to stakeholders. On the production side, I'd want automated pre-period parallel-trends diagnostics that flag geos where the synthetic control fit is poor before the analyst even looks at results.

---

## From Demo to Production

**Q: If Mozilla wanted to run this methodology on real Firefox install data, what would need to change?**

**A:** The core methodological scaffolding transfers directly — synthetic control, permutation tests, and bootstrapped CIs are all data-agnostic. The main adaptation work would be on data sourcing: I'd need to pull actual Firefox download or market share time series (StatCounter, browser telemetry aggregates, or internal install data at the geo level), align campaign spend data from Mozilla's media-buying platforms by geo and period, and establish a clean pre-intervention window. The synthetic data used 36 months; I'd want to verify that the real data has sufficient pre-period length relative to the number of donor geos to produce stable weights — as a rule of thumb, you want more pre-period observations than donor pool members.

**Q: What data sources and label definitions would you validate first?**

**A:** First, I'd align on what "incremental Firefox install" means operationally — is it new installs, active users, market share point change, or something else — because that definition drives everything downstream including how we interpret CPI. Second, I'd audit the geo-level spend data for completeness: are there geos where spend was partially treated (e.g., programmatic leakage across borders) that should be excluded from the donor pool? Third, I'd run the parallel-trends check on the real pre-period before committing to any treatment/control assignment, because if the control geos don't track the treated geo pre-campaign, the synthetic control estimate will be biased and we'd need to expand or rethink the donor pool.

**Q: How would you monitor this in production, and how often would you re-run it?**

**A:** Synthetic control is typically run post-campaign as an end-of-flight measurement rather than in real time, but I'd set up an automated pipeline — similar to the MLOps framework I built at Trend Micro — that ingests new geo-level data, checks pre-period RMSE against a threshold, runs the optimization, and produces a standardized results report with a quality gate before any estimates get surfaced to stakeholders. For ongoing campaigns I'd add a cumulative lift monitor so we can see if lift is plateauing, which is an early signal for budget reallocation. I'd also version the donor pool weights so we can audit how they shift as new data arrives.

---

## Behavioral & Ownership

**Q: How would you explain this methodology and its results to a non-technical marketing stakeholder?**

**A:** I'd lead with the business question and the answer, not the method. Something like: "We ran paid media in Germany, France, and Poland — and we wanted to know whether that spend actually caused more Firefox users, not just whether downloads happened to go up at the same time. To figure that out, we built a 'what would have happened without the campaign' baseline using similar markets that didn't receive spend. Germany showed a 2.8 percentage point lift at $1.08 per incremental install; Poland had the highest percentage lift at 3.1 pp for $1.56. The statistical confidence is very high — none of our non-treated markets showed anything close to that level of movement." I'd use the Streamlit dashboard to let them explore geos interactively rather than walking through every chart in a slide deck, because that tends to drive more genuine engagement and better questions.

**Q: Describe a tradeoff you made in this project and how you'd handle it differently with a real stakeholder team.**

**A:** The clearest tradeoff was using a 12-geo donor pool on 36 months of data — that's actually on the smaller side for synthetic control, and I accepted it to keep the demonstration tractable and the runtime under a minute. With a real stakeholder team I'd have a direct conversation about that tradeoff upfront: a larger donor pool and longer pre-period improve counterfactual quality but require more data investment and potentially longer timelines. I'd also loop in the media planning team before finalizing the geo assignment, because they often have context about partial-market buys or channel mix differences that aren't visible in the aggregate spend data and can invalidate the donor pool if not accounted for. At Trend Micro I learned early that the best models fail when the data assumptions aren't pressure-tested by people who know the operational context.
