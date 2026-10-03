-- 03. Highest-value customers to contact
-- Technique: CTEs, window functions (ROW_NUMBER, NTILE)

WITH scored AS (
    SELECT
        policy_id, product_type, treaty_type,
        round(policy_tenure_months / 12.0, 1)   AS tenure_years,
        coalesce(premium_loading_pct, 0)        AS loading_pct,
        annual_premium, ceded_premium,
          CASE WHEN policy_tenure_months <= 24 THEN 2 ELSE 0 END
        + CASE WHEN coalesce(premium_loading_pct, 0) >= 20 THEN 1 ELSE 0 END
        + CASE WHEN product_type = 'IP' THEN 1 ELSE 0 END  AS risk_score
    FROM policyholders
    WHERE lapsed = 0
),
ranked AS (
    SELECT *,
        risk_score * ceded_premium AS exposure,
        ROW_NUMBER() OVER (ORDER BY risk_score * ceded_premium DESC) AS priority,
        NTILE(10)   OVER (ORDER BY risk_score * ceded_premium DESC) AS decile
    FROM scored
)
SELECT priority, policy_id, product_type, treaty_type,
       tenure_years, loading_pct,
       round(ceded_premium) AS ceded_premium, risk_score
FROM ranked
WHERE priority <= 10
ORDER BY priority;
