-- ============================================================
-- 05. Does underwriting loading drive customers away?
-- Technique: CASE binning, NULL handling, running comparison
-- Question: are loaded customers more likely to cancel, and by how much?
-- ============================================================

WITH banded AS (
    SELECT
        CASE
            WHEN premium_loading_pct IS NULL  THEN 'Not recorded'
            WHEN premium_loading_pct = 0      THEN 'No loading'
            WHEN premium_loading_pct < 20     THEN 'Light (1-19%)'
            WHEN premium_loading_pct < 40     THEN 'Moderate (20-39%)'
            ELSE                                   'Heavy (40%+)'
        END AS loading_band,
        lapsed,
        annual_premium
    FROM policyholders
)
SELECT
    loading_band,
    count(*)                        AS policies,
    round(100.0 * avg(lapsed), 1)   AS lapse_rate_pct,
    round(100.0 * avg(lapsed)
        - (SELECT 100.0 * avg(lapsed) FROM policyholders), 1) AS vs_book_average,
    round(avg(annual_premium))      AS avg_premium
FROM banded
GROUP BY loading_band
ORDER BY lapse_rate_pct DESC;

-- Finding: heavily loaded customers lapse well above the book average.
-- Pricing someone up protects the risk pool but raises the chance they leave.
-- Worth flagging to underwriting, not just to retention.
