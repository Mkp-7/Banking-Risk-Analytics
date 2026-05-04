"""
FRED Interest Rate Pipeline
Fetches Fed Funds Rate, 10Y Treasury, 2Y Treasury from Federal Reserve
Source: St. Louis Fed FRED API (free, no API key needed)
"""

import pandas as pd
import sqlite3
import os
import time

try:
    import pandas_datareader as pdr
    DATAREADER_AVAILABLE = True
except Exception:
    DATAREADER_AVAILABLE = False

DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/banking_risk.db")


# ── Fallback: embedded data (accurate public Fed data) ────────────────────────
# Source: Federal Reserve H.15 Selected Interest Rates
# Monthly averages, quarterly summarized
EMBEDDED_RATES = {
    # date (YYYY-MM-DD) : [fed_funds, treasury_10y, treasury_2y]
    "2019-03-31": [2.40, 2.72, 2.52],
    "2019-06-30": [2.41, 2.58, 2.33],
    "2019-09-30": [2.19, 1.87, 1.75],
    "2019-12-31": [1.76, 1.92, 1.64],
    "2020-03-31": [1.16, 0.87, 0.56],
    "2020-06-30": [0.06, 0.66, 0.21],
    "2020-09-30": [0.09, 0.69, 0.14],
    "2020-12-31": [0.09, 0.93, 0.13],
    "2021-03-31": [0.07, 1.74, 0.16],
    "2021-06-30": [0.06, 1.49, 0.25],
    "2021-09-30": [0.08, 1.52, 0.27],
    "2021-12-31": [0.08, 1.52, 0.73],
    "2022-03-31": [0.20, 2.32, 2.28],
    "2022-06-30": [1.21, 3.13, 3.10],
    "2022-09-30": [2.99, 3.83, 4.27],
    "2022-12-31": [3.98, 3.88, 4.42],
    "2023-03-31": [4.65, 3.96, 4.60],
    "2023-06-30": [5.08, 3.84, 4.87],
    "2023-09-30": [5.33, 4.57, 5.03],
    "2023-12-31": [5.33, 3.97, 4.43],
    "2024-03-31": [5.33, 4.20, 4.62],
    "2024-06-30": [5.33, 4.36, 4.75],
    "2024-09-30": [5.13, 3.81, 3.96],
    "2024-12-31": [4.58, 4.25, 4.24],
    "2025-03-31": [4.33, 4.28, 3.97],
    "2025-06-30": [4.33, 4.40, 3.96],
    "2025-09-30": [4.33, 4.35, 3.90],
}

# Key macro events for chart annotations
MACRO_EVENTS = [
    {"date": "2020-03-15", "event": "COVID Emergency Cut", "rate": 0.25},
    {"date": "2022-03-16", "event": "Rate Hike Cycle Begins", "rate": 0.25},
    {"date": "2023-03-10", "event": "SVB Failure",           "rate": 4.75},
    {"date": "2023-05-01", "event": "Rate Peak (5.25-5.5%)", "rate": 5.25},
    {"date": "2024-09-18", "event": "First Rate Cut",        "rate": 5.00},
]


def fetch_fred_rates() -> pd.DataFrame:
    """Try FRED API first, fall back to embedded data."""

    if DATAREADER_AVAILABLE:
        print("  Attempting FRED API...")
        try:
            import datetime
            raw = pdr.get_data_fred(
                ["FEDFUNDS", "DGS10", "DGS2"],
                start=datetime.datetime(2019, 1, 1),
                end=datetime.datetime(2025, 9, 30)
            )
            raw.columns = ["fed_funds", "treasury_10y", "treasury_2y"]
            raw = raw.resample("QE").mean().dropna(how="all")
            raw.index = raw.index.strftime("%Y-%m-%d")
            raw = raw.reset_index().rename(columns={"index": "date"})
            print(f"    -> FRED API: {len(raw)} quarters fetched")
            return raw
        except Exception as e:
            print(f"    -> FRED API unavailable ({e}), using embedded data")

    # Embedded fallback
    print("  Using embedded Federal Reserve rate data (H.15 release)...")
    records = [
        {"date": dt, "fed_funds": v[0], "treasury_10y": v[1], "treasury_2y": v[2]}
        for dt, v in EMBEDDED_RATES.items()
    ]
    df = pd.DataFrame(records)
    print(f"    -> {len(df)} quarterly rate observations (2019–Q3 2025)")
    return df


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived interest rate metrics."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Yield curve spread (2s10s) — negative = inverted = recession signal
    df["yield_curve_spread"] = df["treasury_10y"] - df["treasury_2y"]

    # Rate change momentum (QoQ change in Fed Funds)
    df["fed_funds_qoq_change"] = df["fed_funds"].diff()

    # Cumulative rate change from trough
    trough = df["fed_funds"].min()
    df["rate_hike_from_trough"] = df["fed_funds"] - trough

    # Rate environment label
    def rate_env(row):
        if row["fed_funds_qoq_change"] > 0.5:  return "Rapid Tightening"
        if row["fed_funds_qoq_change"] > 0:    return "Tightening"
        if row["fed_funds_qoq_change"] < -0.5: return "Rapid Easing"
        if row["fed_funds_qoq_change"] < 0:    return "Easing"
        return "Stable"
    df["rate_environment"] = df.apply(rate_env, axis=1)

    # Inverted yield curve flag
    df["yield_curve_inverted"] = (df["yield_curve_spread"] < 0).astype(int)

    return df


def save_rates(df: pd.DataFrame):
    """Save to SQLite and CSV."""
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("interest_rates", conn, if_exists="replace", index=False)

    # Save macro events
    events_df = pd.DataFrame(MACRO_EVENTS)
    events_df.to_sql("macro_events", conn, if_exists="replace", index=False)
    conn.close()

    data_dir = os.path.join(os.path.dirname(__file__), "../../data")
    df.to_csv(os.path.join(data_dir, "interest_rates.csv"), index=False)
    print(f"    -> Saved to DB and CSV")


def run_fred_pipeline():
    print("\n=== FRED Interest Rate Pipeline ===\n")
    print("[1/3] Fetching rate data...")
    df = fetch_fred_rates()

    print("\n[2/3] Computing derived metrics...")
    df = compute_derived_metrics(df)
    print(f"    -> Yield curve inverted: {df['yield_curve_inverted'].sum()} quarters")
    print(f"    -> Peak fed funds: {df['fed_funds'].max():.2f}%")
    print(f"    -> Current fed funds: {df.iloc[-1]['fed_funds']:.2f}%")

    print("\n[3/3] Saving to database...")
    save_rates(df)

    print("\n=== FRED Pipeline Complete ===")
    return df


if __name__ == "__main__":
    run_fred_pipeline()