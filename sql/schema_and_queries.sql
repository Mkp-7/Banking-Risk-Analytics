-- ============================================================
-- Banking Operational Risk Intelligence Platform
-- SQL Schema & KRI Queries
-- Data Source: FDIC BankFind API (Real Data)
-- ============================================================

-- ── Core Tables ──────────────────────────────────────────────

-- Institutions: all FDIC-insured active banks
-- (populated by fdic_pipeline.py from FDIC API)
CREATE TABLE IF NOT EXISTS institutions (
    cert_id             TEXT PRIMARY KEY,
    bank_name           TEXT,
    city                TEXT,
    state               TEXT,
    ASSET               REAL,   -- Total assets ($thousands)
    DEP                 REAL,   -- Total deposits
    LNLSNET             REAL,   -- Net loans & leases
    NETINC              REAL,   -- Net income
    ROA                 REAL,   -- Return on assets (%)
    ROE                 REAL,   -- Return on equity (%)
    RBCRWAJ             REAL,   -- Risk-based capital ratio (%)
    LNLSDEPR            REAL,   -- Noncurrent loans
    EQTOT               REAL,   -- Total equity
    INTINC              REAL,   -- Interest income
    NONII               REAL,   -- Noninterest income
    NONIX               REAL,   -- Noninterest expense
    -- Engineered KRIs
    loan_to_deposit_ratio   REAL,
    capital_adequacy_ratio  REAL,
    npl_ratio               REAL,
    cost_to_income_ratio    REAL,
    net_interest_margin     REAL,
    risk_tier               TEXT,   -- HIGH / MEDIUM / LOW
    size_bucket             TEXT    -- Systemically Important / Large / Mid-Size / Community / Small
);

-- Historical financials: 2019-2023 for trend analysis
CREATE TABLE IF NOT EXISTS historical_financials (
    REPDTE      TEXT,
    CERT        TEXT,
    INSTNAME    TEXT,
    STNAME      TEXT,
    ASSET       REAL,
    DEP         REAL,
    LNLSNET     REAL,
    NETINC      REAL,
    ROA         REAL,
    ROE         REAL,
    RBCRWAJ     REAL,
    LNLSDEPR    REAL,
    EQTOT       REAL
);

-- Bank failures: full FDIC historical failure list
CREATE TABLE IF NOT EXISTS failures (
    CERT                    TEXT,
    bank_name               TEXT,
    city                    TEXT,
    state_code              TEXT,
    SAVR                    TEXT,
    RESTYPE                 TEXT,
    RESDATE                 TEXT,
    FAILDATE                TEXT,
    insurance_cost_millions REAL,   -- Cost to FDIC insurance fund ($millions)
    QBFASSET                REAL,   -- Assets at failure
    QBFCERT                 TEXT,
    fail_year               INTEGER,
    fail_decade             INTEGER
);

-- ML risk scores: output from risk_engine.py
CREATE TABLE IF NOT EXISTS risk_scores (
    cert_id                 TEXT,
    bank_name               TEXT,
    state                   TEXT,
    failure_probability     REAL,   -- 0-1 probability from Random Forest
    predicted_at_risk       INTEGER,
    risk_score_100          REAL,   -- 0-100 risk score
    risk_cluster_label      TEXT,   -- K-Means cluster label
    flag_low_capital        INTEGER,
    flag_high_npl           INTEGER,
    flag_negative_roa       INTEGER,
    flag_high_ldr           INTEGER,
    total_flags             INTEGER
);

-- ── KRI Summary View ─────────────────────────────────────────

CREATE VIEW IF NOT EXISTS kri_summary AS
SELECT
    state,
    COUNT(*)                                                    AS bank_count,
    ROUND(AVG(ROA), 3)                                          AS avg_roa,
    ROUND(AVG(ROE), 2)                                          AS avg_roe,
    ROUND(AVG(capital_adequacy_ratio), 2)                       AS avg_capital_ratio,
    ROUND(AVG(npl_ratio), 2)                                    AS avg_npl_ratio,
    ROUND(AVG(loan_to_deposit_ratio), 2)                        AS avg_ldr,
    SUM(CASE WHEN risk_tier = 'HIGH'   THEN 1 ELSE 0 END)       AS high_risk_count,
    SUM(CASE WHEN risk_tier = 'MEDIUM' THEN 1 ELSE 0 END)       AS medium_risk_count,
    SUM(CASE WHEN risk_tier = 'LOW'    THEN 1 ELSE 0 END)       AS low_risk_count,
    ROUND(
        100.0 * SUM(CASE WHEN risk_tier = 'HIGH' THEN 1 ELSE 0 END) / COUNT(*), 1
    )                                                           AS high_risk_pct
FROM institutions
GROUP BY state
ORDER BY high_risk_count DESC;

-- ── KRI Analytical Queries ────────────────────────────────────

-- 1. Control Exception Report: Banks breaching Basel III capital floor
SELECT
    bank_name, state, size_bucket,
    capital_adequacy_ratio,
    CASE
        WHEN capital_adequacy_ratio < 8  THEN 'CRITICAL — Below Minimum'
        WHEN capital_adequacy_ratio < 10 THEN 'WARNING — Below Well-Capitalized'
        ELSE 'COMPLIANT'
    END AS capital_control_status,
    ROA, npl_ratio, risk_tier
FROM institutions
WHERE capital_adequacy_ratio < 10
ORDER BY capital_adequacy_ratio ASC
LIMIT 50;

-- 2. Multi-flag Risk Watch List (banks with 3+ KRI breaches)
SELECT
    i.bank_name, i.state, i.size_bucket,
    i.capital_adequacy_ratio, i.npl_ratio, i.ROA,
    i.loan_to_deposit_ratio,
    rs.failure_probability,
    rs.total_flags,
    rs.risk_cluster_label
FROM institutions i
JOIN risk_scores rs ON i.cert_id = rs.cert_id
WHERE rs.total_flags >= 3
ORDER BY rs.failure_probability DESC;

-- 3. State-level KRI Dashboard (mirrors Power BI view)
SELECT * FROM kri_summary WHERE high_risk_pct > 10;

-- 4. Historical failure cost by decade
SELECT
    fail_decade,
    COUNT(*)                                        AS total_failures,
    ROUND(SUM(insurance_cost_millions), 0)          AS total_cost_millions,
    ROUND(AVG(insurance_cost_millions), 1)          AS avg_cost_per_failure,
    MAX(insurance_cost_millions)                    AS largest_failure_cost
FROM failures
WHERE fail_decade IS NOT NULL
GROUP BY fail_decade
ORDER BY fail_decade;

-- 5. Risk-based sampling for control testing (random 5% sample of high-risk)
SELECT
    i.bank_name, i.state, i.size_bucket,
    i.ROA, i.capital_adequacy_ratio, i.npl_ratio,
    rs.failure_probability,
    rs.total_flags
FROM institutions i
JOIN risk_scores rs ON i.cert_id = rs.cert_id
WHERE i.risk_tier = 'HIGH'
ORDER BY RANDOM()
LIMIT (SELECT CAST(COUNT(*) * 0.05 AS INTEGER) FROM institutions WHERE risk_tier = 'HIGH');

-- 6. Institution trend analysis (year-over-year ROA change)
WITH yearly AS (
    SELECT
        CERT,
        INSTNAME,
        REPDTE,
        ROA,
        LAG(ROA) OVER (PARTITION BY CERT ORDER BY REPDTE) AS prev_roa
    FROM historical_financials
)
SELECT
    CERT, INSTNAME, REPDTE,
    ROUND(ROA, 3) AS roa,
    ROUND(prev_roa, 3) AS prev_year_roa,
    ROUND(ROA - prev_roa, 3) AS roa_change,
    CASE WHEN ROA < prev_roa THEN 'DETERIORATING' ELSE 'IMPROVING' END AS trend
FROM yearly
WHERE prev_roa IS NOT NULL
ORDER BY roa_change ASC
LIMIT 20;
