"""
FDIC BankFind API Data Pipeline
Pulls real banking data: institutions, financials, failures
"""

import requests
import pandas as pd
import sqlite3
import time
import os

BASE_URL = "https://banks.data.fdic.gov/api"
DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/banking_risk.db")


# ── Helpers ──────────────────────────────────────────────────────────────────

def fdic_get(endpoint: str, params: dict) -> list:
    """Paginate through FDIC API and return all records."""
    records = []
    params = {**params, "limit": 10000, "offset": 0, "output": "json"}
    while True:
        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        batch = data.get("data", [])
        if not batch:
            break
        records.extend([rec["data"] for rec in batch])
        if len(batch) < params["limit"]:
            break
        params["offset"] += params["limit"]
        time.sleep(0.3)
    return records


def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


# ── Fetch Functions ───────────────────────────────────────────────────────────

def fetch_institutions() -> pd.DataFrame:
    """Active FDIC-insured banks with financial metrics."""
    print("  Fetching institutions...")
    fields = [
        "REPDTE", "CERT", "INSTNAME", "CITY", "STNAME", "ACTIVE",
        "ASSET", "DEP", "LNLSNET", "NETINC", "ROA", "ROE",
        "INTINC", "NONII", "NONIX", "LNLSDEPR", "EQTOT",
        "RBCRWAJ", "INTEXP", "SC", "LNRE", "LNCI", "LNCON"
    ]
    records = fdic_get("financials", {
        "fields": ",".join(fields),
        "filters": "REPDTE:20231231 AND ACTIVE:1",
        "sort_by": "ASSET",
        "sort_order": "DESC"
    })
    df = pd.DataFrame(records)
    print(f"    -> {len(df)} institutions")
    return df


def fetch_failures() -> pd.DataFrame:
    """All FDIC bank failures with cost to insurance fund."""
    print("  Fetching bank failures...")
    fields = [
        "CERT", "NAME", "CITY", "STALP", "SAVR", "RESTYPE",
        "RESDATE", "FAILDATE", "COST", "QBFASSET", "QBFCERT"
    ]
    records = fdic_get("failures", {"fields": ",".join(fields)})
    df = pd.DataFrame(records)
    print(f"    -> {len(df)} historical failures")
    return df


def fetch_historical_financials() -> pd.DataFrame:
    """Multi-year financials for top 500 banks for trend analysis."""
    print("  Fetching historical financials (2019-2023)...")
    fields = [
        "REPDTE", "CERT", "INSTNAME", "STNAME",
        "ASSET", "DEP", "LNLSNET", "NETINC",
        "ROA", "ROE", "RBCRWAJ", "LNLSDEPR", "EQTOT"
    ]
    all_records = []
    for year in ["20191231", "20201231", "20211231", "20221231", "20231231"]:
        records = fdic_get("financials", {
            "fields": ",".join(fields),
            "filters": f"REPDTE:{year} AND ACTIVE:1",
            "sort_by": "ASSET",
            "sort_order": "DESC",
            "limit": 500,
            "offset": 0
        })
        all_records.extend(records)
        print(f"    -> {year}: {len(records)} records")
        time.sleep(0.5)
    return pd.DataFrame(all_records)


# ── Clean & Engineer Features ─────────────────────────────────────────────────

def clean_institutions(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "ASSET", "DEP", "LNLSNET", "NETINC", "ROA", "ROE",
        "INTINC", "NONII", "NONIX", "LNLSDEPR", "EQTOT",
        "RBCRWAJ", "INTEXP", "SC", "LNRE", "LNCI", "LNCON"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived KRIs
    df["loan_to_deposit_ratio"] = (df["LNLSNET"] / df["DEP"].replace(0, pd.NA)) * 100
    df["capital_adequacy_ratio"] = pd.to_numeric(df.get("RBCRWAJ", pd.NA), errors="coerce")
    df["npl_ratio"] = (df["LNLSDEPR"] / df["LNLSNET"].replace(0, pd.NA)) * 100
    df["cost_to_income_ratio"] = (df["NONIX"] / (df["INTINC"] + df["NONII"]).replace(0, pd.NA)) * 100
    df["net_interest_margin"] = (df["INTINC"] / df["ASSET"].replace(0, pd.NA)) * 100

    # Risk tier
    def risk_tier(row):
        score = 0
        if pd.notna(row.get("ROA")) and row["ROA"] < 0:
            score += 3
        elif pd.notna(row.get("ROA")) and row["ROA"] < 0.5:
            score += 1
        if pd.notna(row.get("capital_adequacy_ratio")) and row["capital_adequacy_ratio"] < 8:
            score += 3
        elif pd.notna(row.get("capital_adequacy_ratio")) and row["capital_adequacy_ratio"] < 10:
            score += 1
        if pd.notna(row.get("npl_ratio")) and row["npl_ratio"] > 5:
            score += 2
        if pd.notna(row.get("loan_to_deposit_ratio")) and row["loan_to_deposit_ratio"] > 90:
            score += 1
        if score >= 5:
            return "HIGH"
        elif score >= 2:
            return "MEDIUM"
        return "LOW"

    df["risk_tier"] = df.apply(risk_tier, axis=1)

    # Bank size
    def size_bucket(asset):
        if pd.isna(asset):
            return "Unknown"
        if asset >= 250_000_000:
            return "Systemically Important (>$250B)"
        elif asset >= 10_000_000:
            return "Large ($10B-$250B)"
        elif asset >= 1_000_000:
            return "Mid-Size ($1B-$10B)"
        elif asset >= 100_000:
            return "Community ($100M-$1B)"
        return "Small (<$100M)"

    df["size_bucket"] = df["ASSET"].apply(size_bucket)
    df.rename(columns={"INSTNAME": "bank_name", "STNAME": "state", "CITY": "city", "CERT": "cert_id"}, inplace=True)
    return df


def clean_failures(df: pd.DataFrame) -> pd.DataFrame:
    df["COST"] = pd.to_numeric(df["COST"], errors="coerce")
    df["QBFASSET"] = pd.to_numeric(df["QBFASSET"], errors="coerce")
    df["FAILDATE"] = pd.to_datetime(df["FAILDATE"], errors="coerce")
    df["fail_year"] = df["FAILDATE"].dt.year
    df["fail_decade"] = (df["fail_year"] // 10 * 10).astype("Int64")
    df.rename(columns={"NAME": "bank_name", "STALP": "state_code", "COST": "insurance_cost_millions"}, inplace=True)
    return df


# ── Save to SQLite ─────────────────────────────────────────────────────────────

def save_to_db(institutions: pd.DataFrame, failures: pd.DataFrame, historical: pd.DataFrame):
    print("  Saving to SQLite...")
    conn = get_db()

    institutions.to_sql("institutions", conn, if_exists="replace", index=False)
    failures.to_sql("failures", conn, if_exists="replace", index=False)
    historical.to_sql("historical_financials", conn, if_exists="replace", index=False)

    # KRI summary view
    conn.execute("DROP VIEW IF EXISTS kri_summary")
    conn.execute("""
        CREATE VIEW kri_summary AS
        SELECT
            state,
            COUNT(*) as bank_count,
            ROUND(AVG(ROA), 3) as avg_roa,
            ROUND(AVG(ROE), 2) as avg_roe,
            ROUND(AVG(capital_adequacy_ratio), 2) as avg_capital_ratio,
            ROUND(AVG(npl_ratio), 2) as avg_npl_ratio,
            ROUND(AVG(loan_to_deposit_ratio), 2) as avg_ldr,
            SUM(CASE WHEN risk_tier = 'HIGH' THEN 1 ELSE 0 END) as high_risk_count,
            SUM(CASE WHEN risk_tier = 'MEDIUM' THEN 1 ELSE 0 END) as medium_risk_count,
            SUM(CASE WHEN risk_tier = 'LOW' THEN 1 ELSE 0 END) as low_risk_count
        FROM institutions
        GROUP BY state
        ORDER BY high_risk_count DESC
    """)
    conn.commit()
    conn.close()
    print(f"  Saved to {DB_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline():
    print("\n=== FDIC Banking Risk Pipeline ===\n")

    print("[1/4] Fetching data from FDIC API...")
    institutions_raw = fetch_institutions()
    failures_raw = fetch_failures()
    historical_raw = fetch_historical_financials()

    print("\n[2/4] Cleaning & engineering features...")
    institutions = clean_institutions(institutions_raw)
    failures = clean_failures(failures_raw)
    historical = pd.DataFrame(historical_raw)

    print("\n[3/4] Saving to database...")
    save_to_db(institutions, failures, historical)

    print("\n[4/4] Exporting CSVs...")
    data_dir = os.path.join(os.path.dirname(__file__), "../../data")
    os.makedirs(data_dir, exist_ok=True)
    institutions.to_csv(os.path.join(data_dir, "institutions.csv"), index=False)
    failures.to_csv(os.path.join(data_dir, "failures.csv"), index=False)
    historical.to_csv(os.path.join(data_dir, "historical_financials.csv"), index=False)

    print("\n=== Pipeline Complete ===")
    print(f"  Institutions: {len(institutions):,}")
    print(f"  Failures:     {len(failures):,}")
    print(f"  Historical:   {len(historical):,}")
    return institutions, failures, historical


if __name__ == "__main__":
    run_pipeline()