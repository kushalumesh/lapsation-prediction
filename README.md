# Lapsation Prediction — Life Insurance Policyholder Retention Modelling

End-to-end machine learning project predicting which life insurance policyholders will lapse their policies. Built with XGBoost, SHAP explainability, demographic fairness auditing, and deployed as an interactive Streamlit application.

**🚀 Live Demo:** *coming soon (Streamlit Cloud deployment)*

---

## Business Problem

Life insurance companies lose significant revenue when policyholders cancel coverage before natural policy expiry — a phenomenon called **lapsation**. Predicting which policyholders are at highest risk of lapsing in the coming months allows insurers to prioritise retention outreach and preserve premium revenue.

This project builds a lapsation prediction model on a portfolio of 100,000 synthetic policyholder records parameterised from APRA's 2024 Life Insurance Quarterly Statistics — the official regulatory dataset covering the Australian life insurance industry.

---

## Key Results

*Filled in as project progresses.*

| Metric | Baseline (Logistic Regression) | XGBoost |
|---|---|---|
| AUC-ROC | *tbd* | *tbd* |
| Precision-Recall AUC | *tbd* | *tbd* |
| Recall at 90% precision | *tbd* | *tbd* |

**Class imbalance:** ~91.7% non-lapse, ~8.3% lapse (matches APRA 2024 industry ratio)

---

## Methodology

Follows the **CRISP-DM** framework:

1. **Business Understanding** — Framed as binary classification with heavy class imbalance
2. **Data Preparation** — Synthetic dataset parameterised from APRA 2024 statistics
3. **Exploratory Data Analysis** — Feature distributions, missingness patterns, class imbalance
4. **Modelling** — Logistic regression baseline vs XGBoost
5. **Evaluation** — AUC-ROC, PR-AUC, calibration, confusion matrix on held-out test set
6. **Explainability** — SHAP values (global and local)
7. **Fairness Audit** — Performance across age bands
8. **Deployment** — Streamlit application

---

## Tech Stack

- **Python 3.11**
- **pandas, numpy** — data manipulation
- **scikit-learn** — logistic regression, evaluation metrics
- **XGBoost** — gradient-boosted tree model
- **SHAP** — model explainability
- **Streamlit** — interactive deployment
- **matplotlib, seaborn** — visualisation
- **Jupyter** — exploratory analysis

---

## Project Structure
lapsation-prediction/
├── data/ # Generated data (gitignored)
├── notebooks/ # Jupyter notebooks for EDA and modelling
├── src/ # Python modules
├── models/ # Trained model artefacts (gitignored)
├── app.py # Streamlit application
├── generate_data.py # Synthetic data generator
├── requirements.txt # Package dependencies
└── README.md


---

## How to Reproduce

```bash
# Clone the repository
git clone https://github.com/kushalumesh/lapsation-prediction.git
cd lapsation-prediction

# Create environment
conda create -n lapsation python=3.11 -y
conda activate lapsation
pip install -r requirements.txt

# Generate synthetic data
python generate_data.py

# Run the Streamlit app
streamlit run app.py
```

---

## Author

**Kushal Umesh** — Master of Data Science, University of Queensland
Data Science Intern, Wipro (Resolution Life Australia account)

- [LinkedIn](https://linkedin.com/in/kushalumesh)
- [GitHub](https://github.com/kushalumesh)

---

## Acknowledgements

Data parameters derived from the [APRA Life Insurance Quarterly Statistics (2024)](https://www.apra.gov.au/life-insurance-quarterly-statistics). Project developed during an internship at Wipro on the Resolution Life Australia account with mentorship from Pavan Kondapalli (industry supervisor) and Alan Huang (academic supervisor, University of Queensland).