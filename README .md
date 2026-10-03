# Lapsation Prediction for Life Insurance and Reinsurance

Predicting which life insurance policyholders cancel cover early, explaining why, and pricing what each lapse costs the reinsurance programme. Logistic regression baseline, XGBoost, SHAP explanations, a fairness check, and a Streamlit app.

---

## The problem

Life insurers lose future premium revenue every time a policyholder cancels before their policy naturally ends. This is called **lapsation**. If an insurer can identify who is likely to lapse, it can contact those customers before they leave.

The catch is that lapsation is rare. Only about 8% of policies lapse in a given year, so a model that simply predicts "nobody will lapse" is 92% accurate and completely useless. The real task is ranking policyholders by risk well enough to make a retention campaign worth running.

The cost doesn't stop at the direct book. Most policies are ceded to a reinsurer under a treaty, so roughly **47% of every premium dollar flows through to the reinsurance programme**. When a policy lapses, that ceded premium stops too. This project predicts the lapse, then prices it through the treaty.

## The data

100,000 synthetic records in two layers, parameterised from [APRA's 2024 Life Insurance Quarterly Statistics](https://www.apra.gov.au/life-insurance-quarterly-statistics).

**Direct layer** — policyholder age, gender, smoker status, product (death cover, TPD, income protection, trauma), sum insured, premium, underwriting loading, tenure, prior claims, and the lapse outcome.

**Reinsurance layer** — each policy is ceded under one of two proportional treaties:

| Treaty | How cession works | In this book |
|---|---|---|
| Quota share | Flat percentage of every policy, set by the treaty | 68% of policies, 25–50% ceded |
| Surplus | Insurer retains AUD 300k per life, cedes the excess | 32% of policies, cession rises with policy size |

The overall lapse rate (8.3%) and the decay of lapse risk with tenure both match the published Australian figures.

Synthetic data was used because real policyholder records are personal and confidential. Calibrating to published regulatory statistics keeps the modelling problem realistic without touching anyone's private information. The pipeline retrains on real records unchanged if data access is granted.

## Results

Measured once on a held-out test set of 15,000 policyholders that was never used for training or tuning.

| Model | AUC-ROC | PR-AUC |
|---|---|---|
| Logistic regression (baseline) | 0.643 | 0.143 |
| **XGBoost** | **0.648** | **0.144** |
| Random guessing | 0.500 | 0.084 |

**Reading these numbers.** PR-AUC is the one that matters here. Random guessing gets 0.084, which is just the lapse rate. The model reaches 0.144, so roughly **1.7× better than chance** at concentrating real lapsers near the top of the ranking.

**In reinsurance terms.** On the held-out test book:

| | |
|---|---|
| Ceded premium in the book | AUD 5.71M |
| Ceded premium lost to lapses | AUD 487k (8.5%) |
| Flagged by the model | AUD 154k — **31.6% of lapsing exposure** |
| Policyholders contacted | 2,900 — 19.3% of the book |

Contacting 19% of policyholders reaches 32% of the ceded premium that would otherwise walk out the door — a lift of roughly 1.6× over contacting people at random.

XGBoost only narrowly beats logistic regression. That is a real result, not a disappointing one. It says the relationships in this data are mostly captured by a linear model, and the extra complexity buys very little. Reporting that honestly is more useful than tuning until the gap looks impressive.

## Why reinsurance fields are not model inputs

The dataset carries treaty type, cession percentage and ceded amounts. None of it is given to the model.

A treaty is an arrangement between the insurer and its reinsurer. The policyholder has no visibility of it, so it cannot influence their decision to cancel. Including `cession_pct` would let the model pick up a correlation running through policy size and appear to learn something it hasn't.

The reinsurance layer is applied **after** prediction instead, converting a lapse probability into ceded premium at risk. Predict on behaviour; price the consequence on the treaty.

## What drives lapsation

From the exploratory analysis and confirmed by the model:

- **Policy tenure dominates.** Lapse risk is about 14% in year one and falls to around 4% by year seven.
- **Underwriting loading increases risk.** Policyholders charged a premium loading are more likely to leave.
- **Product type matters.** Income protection lapses most; trauma cover least.
- **Younger policyholders lapse more** than older ones.
- **Prior claimants are stickier.** People who have claimed tend to stay.

## Approach

1. **Data generation** — synthetic policyholders calibrated to APRA statistics
2. **Exploratory analysis** — distributions, missing data, class imbalance, lapse drivers
3. **Baseline** — logistic regression with balanced class weights
4. **Main model** — XGBoost, four configurations compared on validation
5. **Evaluation** — AUC-ROC and PR-AUC, with a decision threshold tuned on validation and the test set used exactly once
6. **Explainability** — SHAP values, globally and per individual
7. **Fairness** — performance compared across age bands
8. **Reinsurance exposure** — model output converted into ceded premium at risk, by treaty
9. **Deployment** — Streamlit app

Preprocessing sits inside a scikit-learn `Pipeline`, so imputation and scaling are fitted on training data only. The test set is touched a single time, at the very end.

## Project structure

```
lapsation-prediction/
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_baseline_model.ipynb   # Logistic regression
│   ├── 03_xgboost_model.ipynb    # XGBoost, test evaluation, fairness
│   └── 04_explainability.ipynb   # SHAP
├── figures/                      # 21 charts produced by the notebooks
├── results/                      # Metrics as JSON
├── app.py                        # Streamlit app
├── generate_data.py              # Synthetic data generator
└── requirements.txt
```

## Run it yourself

```bash
git clone https://github.com/kushalumesh/lapsation-prediction.git
cd lapsation-prediction

conda create -n lapsation python=3.11 -y
conda activate lapsation
pip install -r requirements.txt

python generate_data.py
```

Then run the notebooks in order (01 → 04), and launch the app:

```bash
streamlit run app.py
```

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · matplotlib · seaborn · Streamlit

## Author

**Kushal Umesh** — Master of Data Science, University of Queensland

[GitHub](https://github.com/kushalumesh)
