"""
Synthetic Life Insurance and Reinsurance Dataset Generator
==========================================================

Generates 100,000 synthetic policyholder records for lapsation modelling,
each attached to a reinsurance treaty so the ceded exposure lost to each
lapse can be quantified.

Parameters are taken from APRA 2024 Life Insurance Quarterly Statistics.

Two layers:
  Direct      - policyholder, product, premium, cover, tenure, lapse outcome
  Reinsurance - treaty type, cession percentage, ceded premium and ceded cover

Output: data/policyholders.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
np.random.seed(SEED)

N_RECORDS = 100_000
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "policyholders.parquet"


def generate_ages(n):
    """Age at policy inception: roughly normal, mean 38, std 11."""
    ages = np.random.normal(loc=38.2, scale=11.4, size=n)
    return np.clip(ages, 18, 75).round().astype(int)


def generate_gender(n):
    return np.random.choice(["M", "F"], size=n, p=[0.53, 0.47])


def generate_smoker_status(n):
    """About 15% of Australian adults smoke."""
    return np.random.choice(["Y", "N"], size=n, p=[0.15, 0.85])


def generate_product_type(n):
    return np.random.choice(
        ["Death", "TPD", "IP", "Trauma"], size=n, p=[0.42, 0.25, 0.20, 0.13]
    )


def generate_sum_insured(n, product_type):
    """Log-normal, adjusted by product type."""
    base = np.random.lognormal(mean=12.7, sigma=0.85, size=n)
    adjustments = {"Death": 1.0, "TPD": 0.85, "IP": 0.15, "Trauma": 0.6}
    factors = np.array([adjustments[p] for p in product_type])
    sum_insured = np.clip(base * factors, 50_000, 5_000_000)
    return sum_insured.round(-3).astype(int)


def generate_annual_premium(sum_insured, age, smoker, product_type):
    """Premium driven by cover amount, age, smoking and product."""
    base_rate = np.where(product_type == "Death", 0.0012,
                np.where(product_type == "TPD", 0.0018,
                np.where(product_type == "IP", 0.0065, 0.0022)))
    age_factor = np.clip(1 + (age - 30) * 0.02, 0.7, 3.5)
    smoker_factor = np.where(smoker == "Y", 1.85, 1.0)
    noise = np.random.lognormal(mean=0, sigma=0.15, size=len(sum_insured))
    return (sum_insured * base_rate * age_factor * smoker_factor * noise).round(2)


def generate_premium_loading(n, smoker):
    """Underwriting loading (0-50%), mostly zero; smokers more likely loaded."""
    has_loading = np.random.random(n) < 0.18
    loading = np.zeros(n)
    loading[has_loading] = np.random.uniform(5, 50, size=has_loading.sum())
    smoker_extra = (smoker == "Y") & (np.random.random(n) < 0.30)
    loading[smoker_extra] = np.maximum(
        loading[smoker_extra], np.random.uniform(15, 50, size=smoker_extra.sum())
    )
    return loading.round(1)


def generate_policy_tenure(n):
    """Tenure in months, skewed toward newer policies."""
    tenure = np.random.exponential(scale=42, size=n)
    return np.clip(tenure, 1, 300).round().astype(int)


def generate_prior_claim_count(n):
    return np.random.poisson(lam=0.12, size=n)



# ---------------------------------------------------------------------------
# Reinsurance layer
# ---------------------------------------------------------------------------
# Every direct policy is ceded to a reinsurer under a treaty. Two structures
# are modelled, both proportional:
#
#   Quota share - a fixed percentage of every policy in the treaty is ceded.
#   Surplus     - the insurer keeps a fixed retention and cedes the excess, so
#                 the cession percentage rises with the size of the policy.
#
# The treaty an insurer places a policy into depends on the product and the
# size of the cover, not on the policyholder's behaviour.

RETENTION_LIMIT = 300_000   # AUD retained per life under the surplus treaty

REINSURERS = ["Gen Re", "Hannover Re", "Munich Re", "RGA", "Swiss Re"]


def generate_treaty_type(n, sum_insured):
    """Larger policies are placed into the surplus treaty, smaller ones into quota share."""
    large = sum_insured > RETENTION_LIMIT
    treaty = np.where(large, "Surplus", "Quota Share")
    # a minority of large policies still sit in quota share by treaty agreement
    swap = large & (np.random.random(n) < 0.18)
    treaty[swap] = "Quota Share"
    return treaty


def generate_cession_pct(treaty_type, sum_insured):
    """
    Share of each policy transferred to the reinsurer.

    Quota share: a flat percentage set by the treaty.
    Surplus: whatever sits above the retention limit, so the percentage
             grows with the sum insured and is zero below the retention.
    """
    quota_rate = np.random.choice([25.0, 40.0, 50.0], size=len(sum_insured), p=[0.3, 0.45, 0.25])

    surplus_rate = np.where(
        sum_insured > RETENTION_LIMIT,
        (sum_insured - RETENTION_LIMIT) / sum_insured * 100,
        0.0,
    )
    surplus_rate = np.clip(surplus_rate, 0, 90)

    return np.where(treaty_type == "Surplus", surplus_rate, quota_rate).round(1)


def generate_reinsurer(n, treaty_type):
    """Treaties are placed with a panel of reinsurers."""
    return np.random.choice(REINSURERS, size=n, p=[0.15, 0.2, 0.25, 0.15, 0.25])


def generate_lapsation(df):
    """
    Target variable. Overall lapse rate ~8.3% (APRA industry level), with:
    - much higher lapse risk in the first couple of years
    - higher risk with higher premium loading
    - higher risk for younger policyholders
    - product differences (income protection lapses most)
    - lower risk for people who have claimed before
    """
    n = len(df)
    log_odds = np.full(n, -3.65)  # baseline tuned to hit ~8.3% overall

    tenure_years = df["policy_tenure_months"].values / 12
    log_odds += np.exp(-tenure_years / 2.5) * 1.8
    log_odds += df["premium_loading_pct"].values * 0.015
    log_odds -= (df["age_at_inception"].values - 40) * 0.015
    log_odds += df["product_type"].map(
        {"Death": 0.0, "TPD": 0.2, "IP": 0.45, "Trauma": -0.1}
    ).values
    log_odds -= df["prior_claim_count"].values * 0.3
    log_odds += np.random.normal(0, 0.6, size=n)

    prob = 1 / (1 + np.exp(-log_odds))
    return (np.random.random(n) < prob).astype(int)


def introduce_missing_values(df):
    """Small amounts of realistic missing data."""
    n = len(df)
    df.loc[np.random.random(n) < 0.04, "premium_loading_pct"] = np.nan
    df.loc[np.random.random(n) < 0.02, "smoker_status"] = np.nan
    df.loc[np.random.random(n) < 0.015, "prior_claim_count"] = np.nan
    return df


def main():
    print(f"Generating {N_RECORDS:,} synthetic policyholder records (seed {SEED})...\n")

    ages = generate_ages(N_RECORDS)
    genders = generate_gender(N_RECORDS)
    smokers = generate_smoker_status(N_RECORDS)
    products = generate_product_type(N_RECORDS)
    sum_insured = generate_sum_insured(N_RECORDS, products)
    premiums = generate_annual_premium(sum_insured, ages, smokers, products)
    loadings = generate_premium_loading(N_RECORDS, smokers)
    tenure = generate_policy_tenure(N_RECORDS)
    prior_claims = generate_prior_claim_count(N_RECORDS)

    treaty_type = generate_treaty_type(N_RECORDS, sum_insured)
    cession_pct = generate_cession_pct(treaty_type, sum_insured)
    reinsurer = generate_reinsurer(N_RECORDS, treaty_type)

    df = pd.DataFrame({
        "policy_id": [f"POL{i:07d}" for i in range(1, N_RECORDS + 1)],
        "age_at_inception": ages,
        "gender": genders,
        "smoker_status": smokers,
        "product_type": products,
        "sum_insured": sum_insured,
        "annual_premium": premiums,
        "premium_loading_pct": loadings,
        "policy_tenure_months": tenure,
        "prior_claim_count": prior_claims,
        # reinsurance layer
        "treaty_type": treaty_type,
        "reinsurer": reinsurer,
        "cession_pct": cession_pct,
    })

    # Ceded amounts follow directly from the cession percentage.
    df["ceded_premium"] = (df["annual_premium"] * df["cession_pct"] / 100).round(2)
    df["ceded_sum_insured"] = (df["sum_insured"] * df["cession_pct"] / 100).round(0).astype(int)

    df["lapsed"] = generate_lapsation(df)
    df = introduce_missing_values(df)

    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Shape: {df.shape}\n")
    print("Class balance:")
    print(df["lapsed"].value_counts(normalize=True).round(4))
    print()
    print("Reinsurance layer:")
    print(f"  Total direct premium   AUD {df['annual_premium'].sum():>14,.0f}")
    print(f"  Total ceded premium    AUD {df['ceded_premium'].sum():>14,.0f}"
          f"  ({df['ceded_premium'].sum() / df['annual_premium'].sum():.1%} of direct)")
    print(f"  Ceded premium at risk  AUD {df.loc[df.lapsed == 1, 'ceded_premium'].sum():>14,.0f}"
          f"  (lost to lapses)")
    print()
    print(df.groupby("treaty_type")["cession_pct"].describe()[["count", "mean", "min", "max"]].round(1))


if __name__ == "__main__":
    main()
