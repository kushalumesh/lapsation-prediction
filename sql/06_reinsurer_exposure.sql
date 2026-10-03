-- 06. Exposure by reinsurer
-- Technique: RANK, share of total

SELECT
    reinsurer,
    count(*)                                              AS policies,
    round(sum(ceded_sum_insured) / 1e6, 1)                AS ceded_cover_millions,
    round(sum(ceded_premium))                             AS ceded_premium,
    round(100.0 * sum(ceded_premium) / sum(sum(ceded_premium)) OVER (), 1)
                                                          AS pct_of_ceded_premium,
    round(sum(ceded_premium) FILTER (WHERE lapsed = 1))    AS ceded_lost,
    RANK() OVER (ORDER BY sum(ceded_premium) FILTER (WHERE lapsed = 1) DESC)
                                                          AS exposure_rank
FROM policyholders
GROUP BY reinsurer
ORDER BY exposure_rank;
