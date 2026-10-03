-- ============================================================
-- 02. Premium at risk by reinsurance treaty
-- Technique: conditional aggregation (FILTER), derived ratios
-- Question: how much ceded premium do we lose to lapses, and where?
-- ============================================================

SELECT
    treaty_type,
    count(*)                                               AS policies,
    round(avg(cession_pct), 1)                             AS avg_cession_pct,
    round(sum(annual_premium))                             AS direct_premium,
    round(sum(ceded_premium))                              AS ceded_premium,
    round(sum(ceded_premium) FILTER (WHERE lapsed = 1))     AS ceded_lost_to_lapses,
    round(100.0 * sum(ceded_premium) FILTER (WHERE lapsed = 1)
                / sum(ceded_premium), 2)                   AS pct_of_ceded_lost
FROM policyholders
GROUP BY treaty_type
ORDER BY ceded_lost_to_lapses DESC;

-- Finding: surplus carries a higher average cession, so each lapse there
-- removes more ceded premium than an equivalent quota share lapse.
