"""
generate_data.py
Reproducible synthetic SaaS user lifecycle dataset.
5,000 users with realistic confounding: email treatment is not random
— users with higher engagement are MORE likely to receive the onboarding email.
This means naive ATE is biased upward; propensity matching corrects it.
"""

import numpy as np
import pandas as pd

SEED = 42


def generate_dataset(n: int = 5000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── User features ──────────────────────────────────────────────────────
    days_since_signup = rng.integers(0, 91, size=n).astype(float)
    sessions_last_14d = rng.negative_binomial(n=3, p=0.25, size=n).clip(0, 30).astype(float)
    messages_sent = rng.negative_binomial(n=5, p=0.06, size=n).clip(0, 200).astype(float)

    plan_type = rng.choice(["free", "pro"], size=n, p=[0.70, 0.30])
    industry = rng.choice(
        ["tech", "finance", "healthcare", "other"], size=n, p=[0.40, 0.20, 0.15, 0.25]
    )

    # ── Propensity for email treatment (CONFOUNDED) ────────────────────────
    # More active users are more likely to receive the email (admin targeting bias)
    log_odds_email = (
        -0.5
        + 0.08 * sessions_last_14d
        + 0.004 * messages_sent
        + 0.3 * (plan_type == "pro").astype(float)
        - 0.005 * days_since_signup
        + rng.normal(0, 0.4, size=n)
    )
    p_email = 1 / (1 + np.exp(-log_odds_email))
    received_onboarding_email = rng.binomial(1, p_email).astype(float)

    # ── True conversion probability ────────────────────────────────────────
    # True causal effect of email = +0.08 on log-odds (modest lift)
    log_odds_conv = (
        -3.0
        + 0.15 * sessions_last_14d
        + 0.008 * messages_sent
        + 0.6 * (plan_type == "pro").astype(float)
        + 0.5 * (industry == "tech").astype(float)
        + 0.3 * (industry == "finance").astype(float)
        - 0.01 * days_since_signup
        + 0.4 * received_onboarding_email   # true causal effect
        + rng.normal(0, 0.3, size=n)
    )
    p_conv = 1 / (1 + np.exp(-log_odds_conv))
    converted_to_paid = rng.binomial(1, p_conv).astype(float)

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "days_since_signup": days_since_signup,
            "sessions_last_14d": sessions_last_14d,
            "messages_sent": messages_sent,
            "plan_type": plan_type,
            "industry": industry,
            "received_onboarding_email": received_onboarding_email,
            "converted_to_paid": converted_to_paid,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("saas_users.csv", index=False)
    print(f"Dataset saved: {len(df):,} rows")
    print(df.dtypes)
    print(df.describe())
    print(f"\nConversion rate: {df['converted_to_paid'].mean():.3f}")
    print(f"Email treatment rate: {df['received_onboarding_email'].mean():.3f}")
