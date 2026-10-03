# Lapsation Prediction: Life Insurance Policyholder Retention

Predicting which life insurance policyholders will cancel their cover early, and explaining why. Logistic regression baseline, XGBoost, SHAP explanations, a fairness check, and a Streamlit app.

---

## The problem

Life insurers lose future premium revenue every time a policyholder cancels before their policy naturally ends. This is called **lapsation**. If an insurer can identify who is likely to lapse, it can contact those customers before they leave.

The catch is that lapsation is rare. Only about 8% of policies lapse in a given year, so a model that simply predicts "nobody will lapse" is 92% accurate and completely useless. The real task is ranking policyholders by risk well enough to make a retention campaign worth running.

## The data

100,000 synthetic policyholder records, parameterised from [APRA's 2024 Life Insurance Quarterly Statistics](https://www.apra.gov.au/life-insurance-quarterly-statistics). The overall lapse rate (8.3%) and the way lapse risk falls with policy age both match the published Australian figures.

Synthetic data was used because real policyholder records are personal and confidential. Generating data to match published regulatory statistics keeps the modelling problem realistic without touching anyone's private information.

## Results

Measured once on a held-out test set of 15,000 policyholders that was never used for training or tuning.

| Model | AUC-ROC | PR-AUC |
|---|---|---|
| Logistic regression (baseline) | 0.650 | 0.144 |
| **XGBoost** | **0.654** | **0.146** |
| Random guessing | 0.500 | 0.083 |

**Reading these numbers.** PR-AUC is the one that matters here. Random guessing gets 0.083, which is just the lapse rate. The model reaches 0.146, so roughly **1.75× better than chance** at concentrating real lapsers near the top of the ranking. In campaign terms: contacting people the model flags finds lapsers at nearly double the rate of contacting people at random.

XGBoost only narrowly beats logistic regression. That is a real result, not a disappointing one. It says the relationships in this data are mostly captured by a linear model, and the extra complexity buys very little. Reporting that honestly is more useful than tuning until the gap looks impressive.

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
8. **Deployment** — Streamlit app

Preprocessing sits inside a scikit-learn `Pipeline`, so imputation and scaling are fitted on training data only. The test set is touched a single time, at the very end.

## Project structure

```
lapsation-prediction/
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_baseline_model.ipynb   # Logistic regression
│   ├── 03_xgboost_model.ipynb    # XGBoost, test evaluation, fairness
│   └── 04_explainability.ipynb   # SHAP
├── figures/                      # 20 charts produced by the notebooks
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
