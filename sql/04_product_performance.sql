-- 04. Product performance comparison
-- Technique: window function for percentage of total

SELECT
    product_type,
    count(*)                                           AS policies,
    round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct_of_book,
    round(100.0 * avg(lapsed), 1)                      AS lapse_rate_pct,
    round(avg(annual_premium))                         AS avg_premium,
    round(avg(sum_insured))                            AS avg_sum_insured,
    round(sum(annual_premium) FILTER (WHERE lapsed = 1)) AS premium_lost,
    round(sum(ceded_premium)  FILTER (WHERE lapsed = 1)) AS ceded_lost
FROM policyholders
GROUP BY product_type
ORDER BY lapse_rate_pct DESC;
