# SQL analysis

Six queries against the policyholder dataset, covering the retention and
reinsurance questions the business actually asks.

## Setup

```bash
pip install duckdb
python sql/load_data.py          # loads the parquet into lapsation.duckdb
python sql/run.py                # runs every query
python sql/run.py sql/05_loading_impact.sql   # or just one
```

DuckDB is used because it's file-based, needs no server, and speaks standard
SQL including window functions and CTEs.

## The queries

| File | Technique | Question |
|---|---|---|
| `01_lapse_rate_by_cohort.sql` | CASE binning, aggregation | When do customers leave? |
| `02_premium_at_risk_by_treaty.sql` | Conditional aggregation (FILTER) | How much ceded premium do lapses cost? |
| `03_top_risk_customers.sql` | CTEs, ROW_NUMBER, NTILE | Who should the retention team call? |
| `04_product_performance.sql` | Window function for % of total | Which products lose the most? |
| `05_loading_impact.sql` | CASE binning, NULL handling, subquery | Does pricing people up drive them away? |
| `06_reinsurer_exposure.sql` | RANK, share of total | Which partners carry our lapse exposure? |

## Findings

**Lapse risk is front-loaded.** 13.8% in year one, falling to 4.1% by year
seven. Retention spend on new customers is worth roughly three times the same
spend on long-standing ones.

**Underwriting loading is the clearest single driver.** Heavily loaded
customers lapse at 13.4% against a book average of 8.3%. Loading protects the
risk pool but measurably increases the chance the customer leaves — worth
feeding back to underwriting, not only to retention.

**Lapse rate alone is the wrong target.** Income protection has the highest
rate at 10.8% but the smallest average premium. Death cover lapses less often
yet loses $2.4M against IP's $1.5M. Rate times value is the metric that matters.

**Surplus treaties concentrate the damage.** 43% average cession against 38%
for quota share, so each surplus lapse removes more ceded premium — $1.85M of
the $3.08M lost annually.
