🏦 Banking Operational Risk Intelligence Platform
An end-to-end operational risk analytics system built with real FDIC regulatory data, ML risk scoring, KRI/KPI monitoring, interest rate analysis, and an interactive Streamlit dashboard - replicating workflows used in financial institution compliance and risk management.

🎯 Business Problem
Financial institutions and their regulators need to continuously monitor operational risk across thousands of banks. Manual processes are slow, inconsistent, and fail to catch early warning signs before they become systemic failures. This platform automates:

Control exception detection - automated KRI breach flagging across Basel III thresholds
Risk-based sampling - statistical sampling of high-risk institutions for control testing
Predictive risk scoring - ML models identify at-risk institutions before failure
KRI/KPI reporting - standardized reporting aligned with CCAR and regulatory frameworks
Macro risk overlay - Federal Reserve interest rate analysis and yield curve monitoring


📊 Data Sources
SourceDescriptionAccessFDIC BankFind APIFinancial data for 4,500+ FDIC-insured institutionsFree, no API key requiredFDIC Failures EndpointUS bank failure history since 1934 with insurance costFree, no API key requiredFRED (St. Louis Fed)Fed Funds Rate, 10Y/2Y Treasury yieldsFree via pandas-datareader; embedded fallback included

Note: All data is sourced from US government and Federal Reserve public APIs. Data availability depends on the respective agency's API uptime and rate limits. The FRED pipeline includes embedded fallback data if the API is unavailable.


🏗️ Architecture
banking-risk-platform/
├── src/
│   ├── pipeline/
│   │   ├── fdic_pipeline.py      # FDIC API ingestion, KRI engineering, SQLite load
│   │   └── fred_pipeline.py      # FRED interest rate fetch, yield curve metrics
│   ├── ml/
│   │   └── risk_engine.py        # Logistic Regression, Random Forest, Gradient Boosting, K-Means
│   └── dashboard/
│       └── app.py                # Streamlit executive dashboard
├── sql/
│   └── schema_and_queries.sql    # SQLite schema + 6 analytical KRI queries
├── data/
│   ├── banking_risk.db           # SQLite database
│   ├── institutions.csv
│   ├── failures.csv
│   ├── risk_scores.csv
│   ├── interest_rates.csv
│   └── models/
│       ├── model_metrics.json
│       ├── feature_importance.csv
│       └── cluster_centers.csv
└── run.py                        # Master pipeline runner

🚀 Quick Start
1. Install dependencies
bashpip install -r requirements.txt
2. Run the full pipeline
bashpython run.py
This will:

Pull 4,500+ banks and all historical failures from FDIC API
Fetch Federal Reserve interest rate data from FRED
Train 3 ML models and K-Means clustering
Launch the Streamlit dashboard at http://localhost:8501

3. Run step by step
bash# Phase 1: FDIC data pipeline
python src/pipeline/fdic_pipeline.py

# Phase 2: Interest rate pipeline
python src/pipeline/fred_pipeline.py

# Phase 3: ML risk engine
python src/ml/risk_engine.py

# Phase 4: Dashboard
streamlit run src/dashboard/app.py

Note on rate limits: The FDIC API enforces rate limits. If you encounter a 429 error, the pipeline will retry automatically with increasing wait times. If errors persist, wait 2-3 minutes before re-running.


🤖 Machine Learning
Supervised Models (Binary Classification: At-Risk vs Stable)
ModelROC-AUCAccuracyRecallF1Random Forest0.8471.8%92.3%0.463Gradient Boosting~0.82~70%~89%~0.44Logistic Regression~0.78~68%~85%~0.41
Label Construction: Banks are labeled at-risk based on historical FDIC failure records (1934–present), supplemented by a 30% random sample of currently HIGH-risk institutions. KRI threshold values are intentionally excluded from label construction to avoid circular learning - the same metrics used as features cannot also define the target variable.
Features:

Return on Assets (ROA), Return on Equity (ROE)
Capital Adequacy Ratio (Basel III)
Non-Performing Loan Ratio (NPL)
Loan-to-Deposit Ratio, Net Interest Margin, Cost-to-Income Ratio
Fed Funds Rate, 10Y/2Y Treasury yields, Yield Curve Spread (FRED)

Unsupervised Model (K-Means Clustering)
4 clusters discovered: Low Risk → Elevated Risk → High Risk → Critical Risk

📈 KRI Framework (Basel III / CCAR Aligned)
KRIWarning ThresholdCritical ThresholdRegulatory BasisCapital Adequacy Ratio< 10%< 8%Basel IIINon-Performing Loan Ratio> 5%> 10%FDIC guidanceReturn on Assets< 0.5%< 0%Peer benchmarksLoan-to-Deposit Ratio> 85%> 90%Liquidity riskCost-to-Income Ratio> 60%> 70%Efficiency ratio

🖥️ Dashboard
SectionDescription📊 Executive SummaryKRI cards, risk tier distribution, capital histogram, failure timeline🗺️ Risk HeatmapUS choropleth with traffic-light color coding per KRI metric📈 Macro & RatesFed Funds Rate history, yield curve spread, rate environment analysis🤖 ML Risk ScoringModel comparison, feature importance, high-risk watch list, K-Means clusters✅ Control TestingException counts, breach rates, historical failure analysis, risk-based sampling🔍 Institution LookupSearch any institution by name for full KRI and ML risk profile

🗂️ SQL Queries
sql/schema_and_queries.sql includes 6 production-style KRI queries:

Control Exception Report - Basel III capital breach detection
Multi-Flag Watch List - Banks with 3+ simultaneous KRI breaches
State-Level KRI Summary - Regulatory dashboard view
Historical Failure Cost Analysis - Decade-by-decade failure trends
Risk-Based Sampling - Statistical sampling replicating CCAR examination methodology
YoY Trend Analysis - Institution-level ROA deterioration detection


⚠️ Limitations

Data reflects the most recent FDIC Call Report period available via API (Q3 2025)
SQLite is used for portability; not intended for production-scale concurrent workloads
ML model performance is based on historical failure patterns and may not generalize to novel failure modes
FRED data falls back to embedded quarterly averages if the API is unavailable
