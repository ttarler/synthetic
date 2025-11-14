-- ============================================================================
-- ODFI-RDFI Institutional Relationship Matrix Query
-- ============================================================================
--
-- Purpose: Generate a matrix of transaction volumes between financial
--          institutions (ODFI -> RDFI) for realistic synthetic data generation
--
-- Output: CSV matrix with columns:
--   - odfi: Originating Depository Financial Institution (9-digit routing)
--   - rdfi: Receiving Depository Financial Institution (9-digit routing)
--   - transaction_count: Number of transactions from odfi to rdfi
--   - total_dollar_value: Total dollar volume
--   - avg_dollar_value: Average transaction amount
--   - payment_rail_distribution: JSON of payment rail percentages
--
-- Usage:
--   Run against your existing transaction database and export to CSV
--   Place the output in: config/odfi_rdfi_matrix.csv
--
-- ============================================================================

WITH transaction_volumes AS (
    SELECT
        odfi,
        rdfi,
        COUNT(*) AS transaction_count,
        SUM(dollar_value) AS total_dollar_value,
        AVG(dollar_value) AS avg_dollar_value,

        -- Payment rail distribution (ACH, WIRE, RTP percentages)
        COUNT(CASE WHEN payment_rail = 'ACH' THEN 1 END) * 100.0 / COUNT(*) AS ach_pct,
        COUNT(CASE WHEN payment_rail = 'WIRE' THEN 1 END) * 100.0 / COUNT(*) AS wire_pct,
        COUNT(CASE WHEN payment_rail = 'RTP' THEN 1 END) * 100.0 / COUNT(*) AS rtp_pct,

        -- Recurring payment percentage
        COUNT(CASE WHEN recurring_payment = 'YES' THEN 1 END) * 100.0 / COUNT(*) AS recurring_pct,

        -- Average settlement days (useful for modeling)
        AVG(
            JULIANDAY(rdfi_transaction_settlement_date) -
            JULIANDAY(transaction_create_date)
        ) AS avg_settlement_days

    FROM transactions

    -- Filter conditions (adjust based on your needs)
    WHERE 1=1
        AND odfi IS NOT NULL
        AND rdfi IS NOT NULL
        AND dollar_value > 0
        -- Optional: Filter by date range for recent patterns
        -- AND transaction_create_date >= DATE('now', '-90 days')
        -- Optional: Exclude errors/returns
        -- AND return_/_error_reason = ''

    GROUP BY odfi, rdfi

    -- Filter out very low-volume relationships (noise)
    HAVING transaction_count >= 10
)

SELECT
    odfi,
    rdfi,
    transaction_count,
    ROUND(total_dollar_value, 2) AS total_dollar_value,
    ROUND(avg_dollar_value, 2) AS avg_dollar_value,
    ROUND(ach_pct, 2) AS ach_percentage,
    ROUND(wire_pct, 2) AS wire_percentage,
    ROUND(rtp_pct, 2) AS rtp_percentage,
    ROUND(recurring_pct, 2) AS recurring_percentage,
    ROUND(avg_settlement_days, 2) AS avg_settlement_days,

    -- Calculate relationship strength (normalized score 0-1)
    -- Higher volume and frequency = stronger relationship
    ROUND(
        (transaction_count * 1.0 / MAX(transaction_count) OVER ()) * 0.5 +
        (total_dollar_value * 1.0 / MAX(total_dollar_value) OVER ()) * 0.5,
        4
    ) AS relationship_strength

FROM transaction_volumes

ORDER BY transaction_count DESC, total_dollar_value DESC;

-- ============================================================================
-- Alternative: Aggregate by Institution Pairs (if you need summary stats)
-- ============================================================================

-- Uncomment to get summary statistics about institutional relationships:

-- WITH institution_stats AS (
--     SELECT
--         COUNT(DISTINCT odfi) AS unique_odfi_count,
--         COUNT(DISTINCT rdfi) AS unique_rdfi_count,
--         COUNT(*) AS total_relationship_pairs,
--         SUM(transaction_count) AS total_transactions,
--         SUM(total_dollar_value) AS total_volume,
--         AVG(transaction_count) AS avg_transactions_per_pair,
--         AVG(total_dollar_value) AS avg_volume_per_pair
--     FROM transaction_volumes
-- )
-- SELECT * FROM institution_stats;

-- ============================================================================
-- Export Instructions:
-- ============================================================================
--
-- SQLite:
--   .mode csv
--   .headers on
--   .output config/odfi_rdfi_matrix.csv
--   [run query above]
--   .output stdout
--
-- PostgreSQL:
--   \copy (SELECT ...) TO 'config/odfi_rdfi_matrix.csv' WITH CSV HEADER;
--
-- MySQL:
--   SELECT ... INTO OUTFILE 'config/odfi_rdfi_matrix.csv'
--   FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';
--
-- ============================================================================
