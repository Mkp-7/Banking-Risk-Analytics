"""
Banking Operational Risk Intelligence Platform
Streamlit Dashboard — KRI/KPI Monitoring, Risk Scoring, Control Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Banking Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Resolve absolute paths so Streamlit finds files regardless of working directory
# Resolve project root — walk up from this file until we find the data folder
def _find_project_root():
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):  # search up to 5 levels up
        candidate = os.path.join(current, "data", "banking_risk.db")
        if os.path.exists(candidate):
            return current
        current = os.path.dirname(current)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_ROOT = _find_project_root()
DB_PATH = os.path.join(_ROOT, "data", "banking_risk.db")
MODEL_DIR = os.path.join(_ROOT, "data", "models")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
    .risk-high   { color: #dc2626; font-weight: 700; }
    .risk-medium { color: #d97706; font-weight: 700; }
    .risk-low    { color: #16a34a; font-weight: 700; }
    .kri-card    { background: #1e293b; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_all_data():
    conn = sqlite3.connect(DB_PATH)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
    data = {}
    for t in tables:
        data[t] = pd.read_sql(f"SELECT * FROM {t}", conn)
    conn.close()
    return data


@st.cache_data
def load_metrics():
    path = os.path.join(MODEL_DIR, "model_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_feature_importance():
    path = os.path.join(MODEL_DIR, "feature_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame):
    st.sidebar.image("https://www.fdic.gov/assets/images/fdic-logo.png", width=120)
    st.sidebar.title("🏦 Risk Intelligence")
    st.sidebar.markdown("**Data Source:** FDIC BankFind API")
    st.sidebar.markdown("**Regulation:** CCAR / Basel III / KRI Framework")
    st.sidebar.divider()

    states = sorted(df["state"].dropna().unique().tolist())
    selected_states = st.sidebar.multiselect("Filter by State", states, default=[])

    sizes = df["size_bucket"].dropna().unique().tolist()
    selected_sizes = st.sidebar.multiselect("Bank Size", sizes, default=[])

    risk_tiers = ["HIGH", "MEDIUM", "LOW"]
    selected_risks = st.sidebar.multiselect("Risk Tier", risk_tiers, default=risk_tiers)

    st.sidebar.divider()
    st.sidebar.markdown("**Model Info**")
    metrics = load_metrics()
    if metrics and "Random Forest" in metrics:
        m = metrics["Random Forest"]
        st.sidebar.metric("RF ROC-AUC", f"{m['roc_auc']:.3f}")
        st.sidebar.metric("RF Accuracy", f"{m['accuracy']:.1%}")
        st.sidebar.metric("RF F1 Score", f"{m['f1']:.3f}")

    return selected_states, selected_sizes, selected_risks


def apply_filters(df, states, sizes, risks):
    if states:
        df = df[df["state"].isin(states)]
    if sizes:
        df = df[df["size_bucket"].isin(sizes)]
    if risks:
        df = df[df["risk_tier"].isin(risks)]
    return df


# ── KRI Cards ─────────────────────────────────────────────────────────────────

def render_kri_cards(df: pd.DataFrame):
    st.subheader("📊 Key Risk Indicators (KRI) — Portfolio Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    total = len(df)
    high_risk = (df["risk_tier"] == "HIGH").sum()
    avg_car = df["capital_adequacy_ratio"].median()
    avg_npl = df["npl_ratio"].median()
    avg_roa = df["ROA"].median()
    avg_ldr = df["loan_to_deposit_ratio"].median()

    c1.metric("Total Institutions", f"{total:,}", help="Institutions in filtered view")
    c2.metric("High Risk ⚠️", f"{high_risk:,}",
              delta=f"{high_risk/total*100:.1f}% of portfolio",
              delta_color="inverse")
    c3.metric("Median Capital Ratio", f"{avg_car:.1f}%",
              delta="Basel III min: 8%",
              delta_color="normal" if avg_car >= 10 else "inverse")
    c4.metric("Median NPL Ratio", f"{avg_npl:.2f}%",
              delta="Threshold: 5%",
              delta_color="normal" if avg_npl < 5 else "inverse")
    c5.metric("Median ROA", f"{avg_roa:.2f}%",
              delta="Healthy: >1%",
              delta_color="normal" if avg_roa >= 1 else "inverse")
    c6.metric("Median Loan/Deposit", f"{avg_ldr:.1f}%",
              delta="Threshold: 90%",
              delta_color="normal" if avg_ldr < 90 else "inverse")


# ── Tab: Risk Overview ────────────────────────────────────────────────────────

def tab_overview(df: pd.DataFrame, failures: pd.DataFrame):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Risk Tier Distribution")
        risk_counts = df["risk_tier"].value_counts().reset_index()
        fig = px.pie(risk_counts, values="count", names="risk_tier",
                     color="risk_tier",
                     color_discrete_map={"HIGH": "#dc2626", "MEDIUM": "#f59e0b", "LOW": "#16a34a"},
                     hole=0.4)
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Banks by Size Category")
        size_counts = df["size_bucket"].value_counts().reset_index()
        fig = px.bar(size_counts, x="count", y="size_bucket", orientation="h",
                     color="count", color_continuous_scale="Blues")
        fig.update_layout(height=320, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Capital Ratio vs ROA (Risk Colored)")
        sample = df.sample(min(1000, len(df)), random_state=42)
        sample["npl_size"] = sample["npl_ratio"].fillna(0).clip(lower=0)
        fig = px.scatter(sample, x="capital_adequacy_ratio", y="ROA",
                         color="risk_tier", size="npl_size",
                         color_discrete_map={"HIGH": "#dc2626", "MEDIUM": "#f59e0b", "LOW": "#16a34a"},
                         hover_data=["city", "state"],
                         labels={"capital_adequacy_ratio": "Capital Adequacy Ratio (%)", "ROA": "Return on Assets (%)"},
                         opacity=0.7)
        fig.add_vline(x=8, line_dash="dash", line_color="red", annotation_text="Basel III Min (8%)")
        fig.add_hline(y=0, line_dash="dash", line_color="orange", annotation_text="Break-even ROA")
        fig.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("#### Bank Failures by Year (1934–Present)")
        if "fail_year" in failures.columns:
            yearly = failures.groupby("fail_year").agg(
                failures=("CERT", "count"),
                total_cost=("insurance_cost_millions", "sum")
            ).reset_index()
            yearly = yearly[yearly["fail_year"] >= 1980]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=yearly["fail_year"], y=yearly["failures"],
                                  name="# Failures", marker_color="#3b82f6"), secondary_y=False)
            fig.add_trace(go.Scatter(x=yearly["fail_year"], y=yearly["total_cost"],
                                      name="Cost ($M)", line=dict(color="#dc2626", width=2)), secondary_y=True)
            fig.update_layout(height=340, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)


# ── Tab: KRI Heatmap ──────────────────────────────────────────────────────────

def tab_kri_heatmap(df: pd.DataFrame):
    st.markdown("#### State-Level KRI Heatmap")

    kri_state = df.groupby("state").agg(
        bank_count=("cert_id", "count"),
        avg_capital_ratio=("capital_adequacy_ratio", "median"),
        avg_npl=("npl_ratio", "median"),
        avg_roa=("ROA", "median"),
        high_risk_pct=("risk_tier", lambda x: (x == "HIGH").mean() * 100)
    ).reset_index()

    metric = st.selectbox("KRI Metric", [
        "high_risk_pct", "avg_capital_ratio", "avg_npl", "avg_roa"
    ], format_func=lambda x: {
        "high_risk_pct": "% High Risk Banks",
        "avg_capital_ratio": "Median Capital Adequacy Ratio",
        "avg_npl": "Median NPL Ratio",
        "avg_roa": "Median ROA"
    }[x])

    fig = px.choropleth(
        kri_state,
        locations="state",
        locationmode="USA-states",
        color=metric,
        scope="usa",
        color_continuous_scale="RdYlGn_r" if metric != "avg_roa" else "RdYlGn",
        hover_data=["bank_count", "avg_capital_ratio", "avg_npl", "avg_roa", "high_risk_pct"],
        labels={metric: metric.replace("_", " ").title()}
    )
    fig.update_layout(height=480, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Top 10 States by High-Risk Bank Concentration")
    top_states = kri_state.nlargest(10, "high_risk_pct")
    fig2 = px.bar(top_states, x="state", y="high_risk_pct",
                  color="high_risk_pct", color_continuous_scale="Reds",
                  labels={"high_risk_pct": "% High Risk Banks", "state": "State"})
    fig2.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)


# ── Tab: ML Risk Scoring ──────────────────────────────────────────────────────

def tab_ml_scoring(df: pd.DataFrame):
    col1, col2 = st.columns(2)

    metrics = load_metrics()
    fi = load_feature_importance()

    with col1:
        st.markdown("#### Model Performance Comparison")
        if metrics:
            metrics_df = pd.DataFrame(metrics).T.reset_index()
            metrics_df.columns = ["Model"] + list(metrics_df.columns[1:])
            fig = px.bar(
                metrics_df.melt(id_vars="Model", value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"]),
                x="variable", y="value", color="Model", barmode="group",
                labels={"variable": "Metric", "value": "Score"},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=340, yaxis_range=[0, 1], margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Feature Importance (Random Forest)")
        if not fi.empty:
            fig = px.bar(fi, x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Blues")
            fig.update_layout(height=340, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### High-Risk Bank Watch List")
    if "failure_probability" in df.columns:
        watchlist = df[df["failure_probability"] > 0.5].sort_values(
            "failure_probability", ascending=False
        )[["city", "state", "size_bucket", "ROA", "capital_adequacy_ratio",
           "npl_ratio", "failure_probability", "total_flags"]].head(50)
        watchlist["failure_probability"] = watchlist["failure_probability"].map("{:.1%}".format)
        st.dataframe(watchlist, use_container_width=True, height=400)
    else:
        # Show risk tier based watchlist using KRI data even without ML scores
        st.info("ML risk scores not available — showing KRI-based high risk banks")
        avail = [c for c in ["city","state","size_bucket","ROA",
                             "capital_adequacy_ratio","npl_ratio",
                             "risk_tier","loan_to_deposit_ratio"] if c in df.columns]
        watchlist = df[df["risk_tier"] == "HIGH"].sort_values(
            "capital_adequacy_ratio", ascending=True
        )[avail].head(50)
        st.dataframe(watchlist, use_container_width=True, height=400)


# ── Tab: Control Testing ──────────────────────────────────────────────────────

def tab_control_testing(df: pd.DataFrame, failures: pd.DataFrame):
    st.markdown("#### Control Exception Analysis — KRI Breach Summary")

    # KRI breaches = banks failing regulatory thresholds
    controls = pd.DataFrame({
        "Control": [
            "Capital Adequacy (CAR < 8%)",
            "Capital Adequacy (CAR < 10%)",
            "NPL Ratio (> 5%)",
            "NPL Ratio (> 10%)",
            "Negative ROA",
            "Loan-to-Deposit Ratio (> 90%)",
            "Cost-to-Income (> 70%)"
        ],
        "Type": ["Capital", "Capital", "Credit", "Credit", "Profitability", "Liquidity", "Efficiency"],
        "Exceptions": [
            (df["capital_adequacy_ratio"] < 8).sum(),
            (df["capital_adequacy_ratio"] < 10).sum(),
            (df["npl_ratio"] > 5).sum(),
            (df["npl_ratio"] > 10).sum(),
            (df["ROA"] < 0).sum(),
            (df["loan_to_deposit_ratio"] > 90).sum(),
            (df["cost_to_income_ratio"] > 70).sum(),
        ]
    })
    controls["Exception Rate"] = (controls["Exceptions"] / len(df) * 100).round(2)
    controls["Status"] = controls["Exception Rate"].apply(
        lambda x: "🔴 Critical" if x > 20 else ("🟡 Elevated" if x > 5 else "🟢 Normal")
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(controls, x="Exceptions", y="Control", color="Type", orientation="h",
                     color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(controls[["Control", "Exceptions", "Exception Rate", "Status"]],
                     use_container_width=True, height=380)

    st.markdown("#### Historical Failure Root Cause Analysis")
    if "insurance_cost_millions" in failures.columns:
        decade_summary = failures.groupby("fail_decade").agg(
            failures=("CERT", "count"),
            total_cost=("insurance_cost_millions", "sum")
        ).reset_index().dropna()
        decade_summary["avg_cost"] = (decade_summary["total_cost"] / decade_summary["failures"]).round(1)

        fig2 = px.bar(decade_summary, x="fail_decade", y="failures",
                      text="failures", color="avg_cost",
                      color_continuous_scale="Reds",
                      labels={"fail_decade": "Decade", "failures": "# Failures", "avg_cost": "Avg Cost ($M)"})
        fig2.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)


# ── Tab: Institution Search ───────────────────────────────────────────────────

def tab_search(df: pd.DataFrame):
    st.markdown("#### 🔍 Institution Risk Profile Lookup")
    search = st.text_input("Search by city or state", placeholder="e.g. New York, California, Chicago...")
    if search:
        # Search across city and state columns which exist in actual DB
        mask = pd.Series([False] * len(df), index=df.index)
        for col in ["city", "state"]:
            if col in df.columns:
                mask = mask | df[col].astype(str).str.contains(search, case=False, na=False)
        results = df[mask]
        if len(results) == 0:
            st.warning("No institutions found. Try searching by city (e.g. 'New York') or state (e.g. 'California')")
        else:
            st.success(f"Found {len(results):,} institutions")
            for _, row in results.head(5).iterrows():
                risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(row.get("risk_tier", ""), "⚪")
                with st.expander(f"{risk_color} {row.get('city','Unknown')} — {row.get('state','')}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Capital Adequacy", f"{row.get('capital_adequacy_ratio', 'N/A'):.1f}%" if pd.notna(row.get('capital_adequacy_ratio')) else "N/A")
                    c2.metric("NPL Ratio", f"{row.get('npl_ratio', 'N/A'):.2f}%" if pd.notna(row.get('npl_ratio')) else "N/A")
                    c3.metric("ROA", f"{row.get('ROA', 'N/A'):.2f}%" if pd.notna(row.get('ROA')) else "N/A")
                    c4, c5, c6 = st.columns(3)
                    c4.metric("Risk Tier", row.get("risk_tier", "N/A"))
                    c5.metric("Size", row.get("size_bucket", "N/A"))
                    c6.metric("Loan/Deposit", f"{row.get('loan_to_deposit_ratio', 'N/A'):.1f}%" if pd.notna(row.get('loan_to_deposit_ratio')) else "N/A")


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    st.title("🏦 Banking Operational Risk Intelligence Platform")
    st.markdown("*FDIC BankFind Data · Basel III KRIs · ML Risk Scoring · CCAR-aligned Analytics*")
    st.divider()

    if not os.path.exists(DB_PATH):
        st.error("⚠️ Database not found. Run the data pipeline first:")
        st.code("python src/pipeline/fdic_pipeline.py\npython src/ml/risk_engine.py")
        return

    data = load_all_data()
    institutions = data.get("institutions", pd.DataFrame())

    # Convert all KRI columns to numeric (SQLite stores everything as strings)
    numeric_cols = ["ROA","ROE","capital_adequacy_ratio","npl_ratio",
                    "loan_to_deposit_ratio","net_interest_margin","cost_to_income_ratio",
                    "ASSET","DEP","LNLSNET","NETINC","failure_probability",
                    "risk_score_100","total_flags"]
    for col in numeric_cols:
        if col in institutions.columns:
            institutions[col] = pd.to_numeric(institutions[col], errors="coerce")
    failures = data.get("failures", pd.DataFrame())
    risk_scores = data.get("risk_scores", pd.DataFrame())
    if not risk_scores.empty:
        for col in numeric_cols:
            if col in risk_scores.columns:
                risk_scores[col] = pd.to_numeric(risk_scores[col], errors="coerce")

    # Merge ML scores if available
    if not risk_scores.empty and "failure_probability" in risk_scores.columns:
        # Find common join key between institutions and risk_scores
        inst_key = next((c for c in ["cert_id","CERT","cert"] if c in institutions.columns), None)
        rs_key   = next((c for c in ["cert_id","CERT","cert"] if c in risk_scores.columns), None)
        ml_cols  = ["failure_probability","risk_score_100","risk_cluster_label",
                    "total_flags","predicted_at_risk","flag_low_capital",
                    "flag_high_npl","flag_negative_roa","flag_high_ldr"]
        ml_cols  = [c for c in ml_cols if c in risk_scores.columns]
        if inst_key and rs_key and ml_cols:
            rs_sub = risk_scores[[rs_key] + ml_cols].copy()
            if rs_key != inst_key:
                rs_sub = rs_sub.rename(columns={rs_key: inst_key})
            df = institutions.merge(rs_sub, on=inst_key, how="left", suffixes=("", "_ml"))
        else:
            df = institutions.copy()
    else:
        df = institutions.copy()

    selected_states, selected_sizes, selected_risks = render_sidebar(df)
    df_filtered = apply_filters(df, selected_states, selected_sizes, selected_risks)

    render_kri_cards(df_filtered)
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Risk Overview",
        "🗺️ KRI Heatmap",
        "🤖 ML Scoring",
        "✅ Control Testing",
        "🔍 Institution Lookup"
    ])

    with tab1:
        tab_overview(df_filtered, failures)
    with tab2:
        tab_kri_heatmap(df_filtered)
    with tab3:
        tab_ml_scoring(df_filtered)
    with tab4:
        tab_control_testing(df_filtered, failures)
    with tab5:
        tab_search(df_filtered)


if __name__ == "__main__":
    main()