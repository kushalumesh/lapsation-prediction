-- ============================================================
-- 01. Lapse rate by policy cohort
-- Technique: CASE binning, aggregation, ordered output
-- Question: when in a policy's life do customers leave?
-- ============================================================

SELECT
    CASE
        WHEN policy_tenure_months <= 12  THEN 'Year 1'
        WHEN policy_tenure_months <= 24  THEN 'Year 2'
        WHEN policy_tenure_months <= 36  THEN 'Year 3'
        WHEN policy_tenure_months <= 72  THEN 'Years 4-6'
        ELSE                                  'Year 7+'
    END                                              AS cohort,
    count(*)                                         AS policies,
    sum(lapsed)                                      AS lapses,
    round(100.0 * avg(lapsed), 1)                    AS lapse_rate_pct,
    round(sum(CASE WHEN lapsed = 1 THEN annual_premium END))  AS premium_lost
FROM policyholders
GROUP BY cohort
ORDER BY min(policy_tenure_months);

-- Finding: lapse risk falls from ~14% in year one to ~4% by year seven.
-- Retention effort is worth several times more on new customers.
