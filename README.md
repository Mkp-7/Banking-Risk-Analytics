# 🏦 Banking Operational Risk Intelligence Platform

> **An end-to-end operational risk analytics system** replicating real-world LCD/compliance analyst workflows at financial institutions — built with real FDIC data, ML risk scoring, KRI/KPI monitoring, and an interactive Streamlit dashboard.

---

## 🎯 Business Problem

Financial institutions (and their regulators) need to continuously monitor operational risk across thousands of banks. Manual processes are slow, inconsistent, and fail to catch early warning signs before they become systemic failures. This platform automates:

- **Control exception detection** — automated KRI breach flagging across Basel III thresholds
- **Risk-based sampling** — statistical sampling of high-risk institutions for control testing
- **Predictive risk scoring** — ML models identify at-risk institutions before failure
- **KRI/KPI reporting** — standardized reporting aligned with CCAR and regulatory frameworks

---

## 📊 Data Sources (100% Real, Public Data)

| Source | Description | Access |
|--------|-------------|--------|
| [FDIC BankFind API](https://banks.data.fdic.gov/api) | Financial data for all 4,500+ FDIC-insured institutions | Free, no API key |
| [FDIC Failure List](https://banks.data.fdic.gov/api/failures) | Every US bank failure since 1934 with insurance cost | Free |
| [Fed Reserve DFAST](https://www.federalreserve.gov/supervisionreg/dfa-stress-tests.htm) | CCAR stress test results for 32 major banks | Public PDF |

---

## 🏗️ Architecture

```
banking-risk-platform/
├── src/
│   ├── pipeline/
│   │   └── fdic_pipeline.py      # FDIC API ingestion, feature engineering, SQLite load
│   ├── ml/
│   │   └── risk_engine.py        # Logistic Regression, Random Forest, K-Means clustering
│   └── dashboard/
│       └── app.py                # Streamlit KRI/KPI dashboard
├── sql/
│   └── schema_and_queries.sql    # SQLite schema + 6 analytical KRI queries
├── data/                         # Generated after running pipeline
│   ├── banking_risk.db           # SQLite database
│   ├── institutions.csv
│   ├── failures.csv
│   ├── risk_scores.csv
│   └── models/
│       ├── best_model.pkl
│       ├── model_metrics.json
│       └── feature_importance.csv
└── run.py                        # Master pipeline runner
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python run.py
```

This will:
- Pull ~4,500 banks + all historical failures from FDIC API (~2-3 min)
- Train 3 ML models + K-Means clustering
- Launch the Streamlit dashboard at `http://localhost:8501`

### 3. Or run step by step
```bash
# Phase 1: Data pipeline
python src/pipeline/fdic_pipeline.py

# Phase 2: ML risk engine
python src/ml/risk_engine.py

# Phase 3: Dashboard
streamlit run src/dashboard/app.py
```

---

## 🤖 Machine Learning

### Supervised Models (Binary Classification: At-Risk vs Stable)

| Model | ROC-AUC | Accuracy | F1 |
|-------|---------|----------|----|
| Random Forest | ~0.92 | ~87% | ~0.84 |
| Gradient Boosting | ~0.91 | ~86% | ~0.83 |
| Logistic Regression | ~0.88 | ~83% | ~0.79 |

**Label Construction:** A bank is labeled "at-risk" if it historically failed (matched to FDIC failure list) OR currently breaches critical KRI thresholds (negative ROA, CAR < 8%, NPL > 10%).

**Features used:**
- Return on Assets (ROA)
- Return on Equity (ROE)
- Capital Adequacy Ratio (Basel III)
- Non-Performing Loan Ratio (NPL)
- Loan-to-Deposit Ratio
- Net Interest Margin
- Cost-to-Income Ratio

### Unsupervised Model (K-Means Clustering)
4 risk clusters: Low Risk → Elevated Risk → High Risk → Critical Risk

---

## 📈 KRI Framework (Basel III / CCAR Aligned)

| KRI | Warning Threshold | Critical Threshold | Regulatory Basis |
|-----|------------------|--------------------|------------------|
| Capital Adequacy Ratio | < 10% | < 8% | Basel III |
| Non-Performing Loan Ratio | > 5% | > 10% | FDIC guidance |
| Return on Assets | < 0.5% | < 0% | Peer benchmarks |
| Loan-to-Deposit Ratio | > 85% | > 90% | Liquidity risk |
| Cost-to-Income Ratio | > 60% | > 70% | Efficiency ratio |

---

## 🖥️ Dashboard Features

| Tab | Description |
|-----|-------------|
| 📈 Risk Overview | Risk tier distribution, scatter plots, failure history timeline |
| 🗺️ KRI Heatmap | Interactive US choropleth — filter by any KRI metric |
| 🤖 ML Scoring | Model comparison, feature importance, high-risk watchlist |
| ✅ Control Testing | Control exception counts, breach rates, decade failure analysis |
| 🔍 Institution Lookup | Search any bank by name for full risk profile |

---

## 🗂️ SQL Queries

`sql/schema_and_queries.sql` includes 6 production-style KRI queries:

1. **Control Exception Report** — Basel III capital breach detection
2. **Multi-Flag Watch List** — Banks with 3+ simultaneous KRI breaches
3. **State-Level KRI Summary** — Regulatory dashboard view
4. **Historical Failure Cost Analysis** — Decade-by-decade failure trends
5. **Risk-Based Sampling** — Statistical sampling for control testing (replicates CCAR methodology)
6. **YoY Trend Analysis** — Institution-level ROA deterioration detection

---