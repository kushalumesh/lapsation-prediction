"""
Synthetic Life Insurance Policyholder Dataset Generator
=======================================================

Generates 100,000 realistic synthetic policyholder records for lapsation
modelling, parameterised from APRA 2024 Life Insurance Quarterly Statistics.

Output: data/policyholders.parquet
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Configuration
N_RECORDS = 100_000
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "policyholders.parquet"


def generate_ages(n):
    """Age at policy inception — approximately normal, mean 38, std 11."""
    ages = np.random.normal(loc=38.2, scale=11.4, size=n)
    ages = np.clip(ages, 18, 75).round().astype(int)
    return ages


def generate_gender(n):
    """Realistic gender split for Australian life insurance."""
    return np.random.choice(["M", "F"], size=n, p=[0.53, 0.47])


def generate_smoker_status(n):
    """~15% of Australian adults are smokers (ABS 2023)."""
    return np.random.choice(["Y", "N"], size=n, p=[0.15, 0.85])


def generate_product_type(n):
    """Product mix from APRA industry data."""
    return np.random.choice(
        ["Death", "TPD", "IP", "Trauma"],
        size=n,
        p=[0.42, 0.25, 0.20, 0.13]
    )


def generate_sum_insured(n, product_type):
    """Log-normal distribution — mean AUD ~485k, std ~612k. Varies by product."""
    # Base log-normal parameters
    mu, sigma = 12.7, 0.85
    base = np.random.lognormal(mean=mu, sigma=sigma, size=n)

    # Product-type adjustments
    adjustments = {
        "Death": 1.0,
        "TPD": 0.85,
        "IP": 0.15,  # Income protection is typically smaller sum
        "Trauma": 0.6,
    }
    factors = np.array([adjustments[p] for p in product_type])
    sum_insured = base * factors

    # Realistic caps and rounding
    sum_insured = np.clip(sum_insured, 50_000, 5_000_000)
    return sum_insured.round(-3).astype(int)


def generate_annual_premium(sum_insured, age, smoker, product_type):
    """Premium calculated from sum insured, age, smoker status, and product."""
    # Base rate per thousand sum insured
    base_rate = np.where(
        product_type == "Death", 0.0012,
        np.where(product_type == "TPD", 0.0018,
        np.where(product_type == "IP", 0.0065,
        0.0022))  # Trauma
    )

    # Age loading
    age_factor = 1 + (age - 30) * 0.02
    age_factor = np.clip(age_factor, 0.7, 3.5)

    # Smoker loading
    smoker_factor = np.where(smoker == "Y", 1.85, 1.0)

    # Random variation
    noise = np.random.lognormal(mean=0, sigma=0.15, size=len(sum_insured))

    premium = sum_insured * base_rate * age_factor * smoker_factor * noise
    return premium.round(2)


def generate_premium_loading(n, smoker):
    """Loading factor applied at underwriting (0-50%, skewed toward 0)."""
    # Most policies have no loading
    has_loading = np.random.random(n) < 0.18
    loading = np.zeros(n)
    loading[has_loading] = np.random.uniform(5, 50, size=has_loading.sum())

    # Smokers more likely to have loading
    smoker_extra = (smoker == "Y") & (np.random.random(n) < 0.30)
    loading[smoker_extra] = np.maximum(loading[smoker_extra], np.random.uniform(15, 50, size=smoker_extra.sum()))

    return loading.round(1)


def generate_policy_tenure(n):
    """Tenure in months. Exponential-like distribution."""
    tenure = np.random.exponential(scale=42, size=n)
    tenure = np.clip(tenure, 1, 300).round().astype(int)
    return tenure


def generate_prior_claim_count(n):
    """Poisson-distributed prior claim count. Most have zero."""
    return np.random.poisson(lam=0.12, size=n)


def generate_lapsation(df):
    """
    Generate lapsation target variable.
    
    Base lapse rate ~8.3% (matching APRA), with dependencies:
    - Higher in first 24 months (exponential decay pattern)
    - Higher for high premium loadings
    - Higher for younger policyholders
    - Product-type effects
    """
    n = len(df)

    # Base rate
    log_odds = np.full(n, -2.5)  # baseline ~7.6% probability

    # Tenure effect — sharply higher early lapsation
    tenure_years = df["policy_tenure_months"].values / 12
    log_odds += np.exp(-tenure_years / 2.5) * 1.8

    # Premium loading effect
    log_odds += df["premium_loading_pct"].values * 0.015

    # Age effect — younger more likely to lapse
    log_odds -= (df["age_at_inception"].values - 40) * 0.015

    # Product-type effect
    product_effect = df["product_type"].map({
        "Death": 0.0,
        "TPD": 0.2,
        "IP": 0.45,   # IP has highest lapse rate historically
        "Trauma": -0.1,
    }).values
    log_odds += product_effect

    # Prior claims — those with prior claims less likely to lapse (they've used the product)
    log_odds -= df["prior_claim_count"].values * 0.3

    # Convert to probability with noise
    log_odds += np.random.normal(0, 0.6, size=n)
    prob = 1 / (1 + np.exp(-log_odds))

    lapsed = (np.random.random(n) < prob).astype(int)
    return lapsed


def introduce_missing_values(df):
    """Realistic missing data patterns (~3-5% in select fields)."""
    n = len(df)

    # premium_loading missing when no loading was assessed
    mask = np.random.random(n) < 0.04
    df.loc[mask, "premium_loading_pct"] = np.nan

    # smoker_status occasionally missing (data quality issue)
    mask = np.random.random(n) < 0.02
    df.loc[mask, "smoker_status"] = np.nan

    # prior_claim_count occasionally missing
    mask = np.random.random(n) < 0.015
    df.loc[mask, "prior_claim_count"] = np.nan

    return df


def main():
    print(f"Generating {N_RECORDS:,} synthetic policyholder records...")
    print(f"Random seed: {SEED}")
    print()

    # Generate features
    policy_ids = [f"POL{str(i).zfill(7)}" for i in range(1, N_RECORDS + 1)]
    ages = generate_ages(N_RECORDS)
    genders = generate_gender(N_RECORDS)
    smokers = generate_smoker_status(N_RECORDS)
    products = generate_product_type(N_RECORDS)
    sum_insured = generate_sum_insured(N_RECORDS, products)
    premiums = generate_annual_premium(sum_insured, ages, smokers, products)
    loadings = generate_premium_loading(N_RECORDS, smokers)
    tenure = generate_policy_tenure(N_RECORDS)
    prior_claims = generate_prior_claim_count(N_RECORDS)

    # Assemble dataframe
    df = pd.DataFrame({
        "policy_id": policy_ids,
        "age_at_inception": ages,
        "gender": genders,
        "smoker_status": smokers,
        "product_type": products,
        "sum_insured": sum_insured,
        "annual_premium": premiums,
        "premium_loading_pct": loadings,
        "policy_tenure_months": tenure,
        "prior_claim_count": prior_claims,
    })

    # Generate target
    df["lapsed"] = generate_lapsation(df)

    # Introduce realistic missing values
    df = introduce_missing_values(df)

    # Save to parquet
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    # Summary
    print(f"Dataset saved to: {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")
    print()
    print("Class balance:")
    print(df["lapsed"].value_counts(normalize=True).round(4))
    print()
    print("Summary statistics:")
    print(df.describe(include="all").round(2))


if __name__ == "__main__":
    main()