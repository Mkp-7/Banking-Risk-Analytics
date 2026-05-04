"""
Banking Operational Risk Intelligence Platform
Morgan Stanley Brand — Executive Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, json, os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="MS Risk Intelligence", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

def _find_root():
    cur = os.path.abspath(__file__)
    for _ in range(6):
        cur = os.path.dirname(cur)
        if os.path.isdir(os.path.join(cur, "data")):
            return cur
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROOT      = _find_root()
DB_PATH   = os.path.join(ROOT, "data", "banking_risk.db")
MODEL_DIR = os.path.join(ROOT, "data", "models")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Morgan Stanley Brand Colors ── */
:root {
  --ms-navy:    #002B51;
  --ms-navy2:   #003A6B;
  --ms-navy3:   #004080;
  --ms-blue:    #187ABA;
  --ms-blue80:  #4695C8;
  --ms-blue40:  #A3CAE3;
  --ms-blue20:  #D1E4F1;
  --ms-jade:    #00857A;
  --ms-jade40:  #99CCCA;
  --ms-white:   #FFFFFF;
  --ms-gray1:   #F4F6F8;
  --ms-gray2:   #E8ECF0;
  --ms-gray3:   #C8D0D8;
  --ms-gray4:   #8A9AAA;
  --ms-gray5:   #4A5A6A;
  --ms-text:    #FFFFFF;
  --ms-text2:   #D1E4F1;
  --ms-text3:   #8AACCC;
  --ms-red:     #C8382A;
  --ms-amber:   #E07820;
  --ms-green:   #00857A;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--ms-navy) !important;
  color: var(--ms-white) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--ms-navy2) !important;
  border-right: 1px solid rgba(70,149,200,0.3) !important;
}
[data-testid="stSidebar"] * { color: var(--ms-white) !important; }
[data-testid="stSidebar"] .stMarkdown p {
  color: var(--ms-text2) !important;
  font-size: .85rem !important;
}

/* ── Tabs as solid buttons ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: var(--ms-navy2) !important;
  border-radius: 8px !important;
  padding: 4px !important;
  gap: 3px !important;
  border: 1px solid rgba(70,149,200,0.35) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: var(--ms-navy3) !important;
  border-radius: 6px !important;
  color: var(--ms-blue40) !important;
  font-weight: 500 !important;
  font-size: .82rem !important;
  letter-spacing: .02em !important;
  padding: .5rem 1.15rem !important;
  border: 1px solid rgba(70,149,200,0.15) !important;
  transition: all .18s !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
  background: rgba(24,122,186,0.25) !important;
  border-color: var(--ms-blue80) !important;
  color: var(--ms-white) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  background: var(--ms-blue) !important;
  color: var(--ms-white) !important;
  font-weight: 700 !important;
  border-color: var(--ms-blue) !important;
  box-shadow: 0 3px 12px rgba(24,122,186,0.5) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
  background: transparent !important;
  padding: 1.2rem 0 !important;
}

/* ── Metrics ── */
[data-testid="stMetricValue"] {
  font-family: 'Libre Baskerville', serif !important;
  font-size: 1.9rem !important;
  font-weight: 700 !important;
  color: var(--ms-white) !important;
}
[data-testid="stMetricLabel"] {
  color: var(--ms-blue40) !important;
  font-size: .68rem !important;
  letter-spacing: .1em !important;
  text-transform: uppercase !important;
}
[data-testid="stMetricDelta"] { font-size: .73rem !important; }

/* ── KRI Cards ── */
.kri-card {
  background: linear-gradient(145deg, var(--ms-navy2), var(--ms-navy3));
  border: 1px solid rgba(70,149,200,0.3);
  border-radius: 10px;
  padding: 1.1rem 1.3rem;
  position: relative;
  overflow: hidden;
  height: 100%;
}
.kri-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 3px; height: 100%;
  background: var(--ms-blue);
}
.kri-card .kc-label {
  font-size: .65rem;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--ms-blue40);
  margin-bottom: .45rem;
  font-weight: 500;
}
.kri-card .kc-val {
  font-family: 'Libre Baskerville', serif;
  font-size: 1.95rem;
  font-weight: 700;
  color: var(--ms-white);
  line-height: 1;
}
.kri-card .kc-sub {
  font-size: .7rem;
  color: var(--ms-text3);
  margin-top: .4rem;
}
.kri-card .kc-bar {
  height: 3px;
  border-radius: 2px;
  margin-top: .75rem;
  background: rgba(255,255,255,0.08);
  overflow: hidden;
}
.kri-card .kc-fill {
  height: 100%;
  border-radius: 2px;
}

/* ── Section titles ── */
.section-title {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: .78rem;
  font-weight: 600;
  color: var(--ms-blue40);
  letter-spacing: .12em;
  text-transform: uppercase;
  padding-bottom: .5rem;
  border-bottom: 1px solid rgba(70,149,200,0.25);
  margin-bottom: .9rem;
}

/* ── Divider ── */
.ms-line {
  height: 1px;
  background: linear-gradient(90deg, var(--ms-blue) 0%, rgba(24,122,186,0.1) 60%, transparent);
  margin: 1.4rem 0;
}

/* ── Badges ── */
.badge { display:inline-block; padding:2px 9px; border-radius:3px;
         font-size:.67rem; font-weight:600; letter-spacing:.05em; }
.badge-high   { background:rgba(200,56,42,.2);  color:#FF6B5A; border:1px solid rgba(200,56,42,.4); }
.badge-medium { background:rgba(181,150,90,.2);  color:#D4AF6A; border:1px solid rgba(181,150,90,.4); }
.badge-low    { background:rgba(0,168,168,.2);  color:#40C8C8; border:1px solid rgba(0,168,168,.4); }

/* ── Hero ── */
.hero-title {
  font-family: 'Libre Baskerville', serif;
  font-size: 2.2rem;
  font-weight: 700;
  color: var(--ms-white);
  line-height: 1.2;
}
.hero-meta {
  font-size: .72rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--ms-blue80);
  margin-top: .3rem;
}

/* ── Inputs ── */
.stTextInput input {
  background: var(--ms-navy2) !important;
  border: 1px solid rgba(70,149,200,0.4) !important;
  color: var(--ms-white) !important;
  border-radius: 6px !important;
  font-size: .88rem !important;
}
.stTextInput input:focus {
  border-color: var(--ms-blue) !important;
  box-shadow: 0 0 0 2px rgba(24,122,186,0.25) !important;
}
.stTextInput label { color: var(--ms-blue40) !important; font-size:.8rem !important; }

/* ── Selects ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
  background: var(--ms-navy2) !important;
  border: 1px solid rgba(70,149,200,0.35) !important;
  border-radius: 6px !important;
  color: var(--ms-white) !important;
}
.stMultiSelect [data-baseweb="tag"] {
  background: rgba(24,122,186,0.3) !important;
  color: var(--ms-blue20) !important;
}
.stSelectbox label, .stMultiSelect label {
  color: var(--ms-blue40) !important;
  font-size: .78rem !important;
  font-weight: 500 !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--ms-blue) !important;
  color: var(--ms-white) !important;
  font-weight: 600 !important;
  border: none !important;
  border-radius: 6px !important;
  letter-spacing: .03em !important;
  transition: all .18s !important;
}
.stButton > button:hover {
  background: var(--ms-blue80) !important;
  box-shadow: 0 4px 14px rgba(24,122,186,0.4) !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
  background: var(--ms-navy2) !important;
  border: 1px solid rgba(70,149,200,0.25) !important;
  border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
  color: var(--ms-white) !important;
}

/* ── DataFrames ── */
.stDataFrame { border: 1px solid rgba(70,149,200,0.2) !important; border-radius: 8px !important; }

/* ── Sidebar radio ── */
.stRadio label {
  background: var(--ms-navy3) !important;
  border: 1px solid rgba(70,149,200,0.15) !important;
  border-radius: 6px !important;
  padding: .4rem .8rem !important;
  color: var(--ms-blue20) !important;
  margin-bottom: 2px !important;
}

/* ── Global text ── */
div[data-testid="stMarkdownContainer"] p { color: var(--ms-text2) !important; }
.stCaption { color: var(--ms-text3) !important; }
h1,h2,h3,h4 { color: var(--ms-white) !important; font-family: 'Libre Baskerville', serif !important; }
</style>
""", unsafe_allow_html=True)

# ── Plot theme (MS colors) ────────────────────────────────────────────────────
def pl(**kw):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,14,35,0.5)",
        font=dict(family="IBM Plex Sans", color="#A3CAE3", size=11),
        margin=dict(t=38, b=28, l=8, r=8),
        xaxis=dict(gridcolor="rgba(70,149,200,0.1)", linecolor="rgba(70,149,200,0.15)",
                   tickfont=dict(color="#A3CAE3", size=10), title_font=dict(color="#A3CAE3")),
        yaxis=dict(gridcolor="rgba(70,149,200,0.1)", linecolor="rgba(70,149,200,0.15)",
                   tickfont=dict(color="#A3CAE3", size=10), title_font=dict(color="#A3CAE3")),
    )
    base.update(kw)
    return base

MS_RISK   = {"HIGH": "#E03030", "MEDIUM": "#F0C020", "LOW": "#28B060"}
MS_BLUES  = ["#D1E4F1", "#A3CAE3", "#4695C8", "#187ABA", "#002B51"]

# Extended chart palette
CHART_PALETTE = ["#187ABA","#C8992A","#1A7A4A","#00857A","#E07820","#6A4BC8","#C8382A"]

# Vivid multi-color heatmap scales per KRI metric
HEATMAP_SCALES = {
    "high_pct": [[0,"#0D0221"],[0.2,"#3B0F70"],[0.4,"#8C2981"],[0.6,"#DE4968"],[0.8,"#FE9F6D"],[1,"#FCFDBF"]],
    "avg_car":  [[0,"#9B0E0E"],[0.25,"#D94040"],[0.45,"#F0A030"],[0.65,"#C8D420"],[0.85,"#40C870"],[1,"#00857A"]],
    "avg_npl":  [[0,"#004050"],[0.2,"#0090A0"],[0.45,"#40C8C0"],[0.65,"#C8A020"],[0.82,"#E06020"],[1,"#C02020"]],
    "avg_roa":  [[0,"#8B0000"],[0.2,"#C83820"],[0.4,"#E07820"],[0.6,"#A8C020"],[0.8,"#30B060"],[1,"#006040"]],
}
# Extended palette - complementary finance-grade colors
CHART_AMBER = "#B5965A"
CHART_AMBER2 = "#D4AF6A"
CHART_TEAL = "#00A8A8"
CHART_TEAL2 = "#40C8C8"
CHART_RED = "#C8382A"
VANG_RED2 = "#8B1A2A"   # Vanguard deep red
STATE_MAP = {
    "ALABAMA":"AL","ALASKA":"AK","ARIZONA":"AZ","ARKANSAS":"AR","CALIFORNIA":"CA",
    "COLORADO":"CO","CONNECTICUT":"CT","DELAWARE":"DE","FLORIDA":"FL","GEORGIA":"GA",
    "HAWAII":"HI","IDAHO":"ID","ILLINOIS":"IL","INDIANA":"IN","IOWA":"IA",
    "KANSAS":"KS","KENTUCKY":"KY","LOUISIANA":"LA","MAINE":"ME","MARYLAND":"MD",
    "MASSACHUSETTS":"MA","MICHIGAN":"MI","MINNESOTA":"MN","MISSISSIPPI":"MS",
    "MISSOURI":"MO","MONTANA":"MT","NEBRASKA":"NE","NEVADA":"NV",
    "NEW HAMPSHIRE":"NH","NEW JERSEY":"NJ","NEW MEXICO":"NM","NEW YORK":"NY",
    "NORTH CAROLINA":"NC","NORTH DAKOTA":"ND","OHIO":"OH","OKLAHOMA":"OK",
    "OREGON":"OR","PENNSYLVANIA":"PA","RHODE ISLAND":"RI","SOUTH CAROLINA":"SC",
    "SOUTH DAKOTA":"SD","TENNESSEE":"TN","TEXAS":"TX","UTAH":"UT",
    "VERMONT":"VT","VIRGINIA":"VA","WASHINGTON":"WA","WEST VIRGINIA":"WV",
    "WISCONSIN":"WI","WYOMING":"WY","DISTRICT OF COLUMBIA":"DC",
    "PUERTO RICO":"PR","VIRGIN ISLANDS":"VI","GUAM":"GU"
}

def fmt(v, d=2):
    try: return f"{float(v):.{d}f}%"
    except: return "N/A"

NUM_COLS = ["ROA","ROE","capital_adequacy_ratio","npl_ratio","loan_to_deposit_ratio",
            "net_interest_margin","cost_to_income_ratio","ASSET","DEP","LNLSNET",
            "NETINC","failure_probability","risk_score_100","total_flags"]

def to_num(df):
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_rates():
    """Load interest rate data from DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rates  = pd.read_sql("SELECT * FROM interest_rates ORDER BY date", conn)
        events = pd.read_sql("SELECT * FROM macro_events", conn)
        conn.close()
        rates["date"] = pd.to_datetime(rates["date"])
        return rates, events
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=300)
def load_all():
    conn = sqlite3.connect(DB_PATH)
    tbls = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
    out  = {t: pd.read_sql(f"SELECT * FROM {t}", conn) for t in tbls}
    conn.close()
    return out

@st.cache_data
def load_metrics():
    p = os.path.join(MODEL_DIR, "model_metrics.json")
    return json.load(open(p)) if os.path.exists(p) else {}

@st.cache_data
def load_fi():
    p = os.path.join(MODEL_DIR, "feature_importance.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar(df):
    st.sidebar.markdown("""
    <div style="padding:.6rem 0 .3rem">
      <div style="font-family:'Libre Baskerville',serif;font-size:1.1rem;
           font-weight:700;color:#FFFFFF;letter-spacing:.02em">
        RISK INTELLIGENCE
      </div>
      <div style="font-size:.6rem;letter-spacing:.18em;color:#4695C8;
           text-transform:uppercase;margin-top:4px">
        Operational Command Center
      </div>
    </div>
    <div style="height:1px;background:rgba(70,149,200,0.4);margin:.7rem 0 1rem"></div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div style="font-size:.65rem;letter-spacing:.12em;color:#4695C8;
         text-transform:uppercase;margin-bottom:.5rem;font-weight:600">
    Data Source
    </div>
    <div style="font-size:.8rem;color:#D1E4F1;line-height:1.9">
      📊 &nbsp;FDIC BankFind API<br>
      📅 &nbsp;September 30, 2025<br>
      📋 &nbsp;FDIC Call Report Q3 2025<br>
      ⚖️ &nbsp;Basel III / CCAR Framework
    </div>
    <div style="height:1px;background:rgba(70,149,200,0.3);margin:.9rem 0"></div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div style="font-size:.65rem;letter-spacing:.12em;color:#4695C8;
         text-transform:uppercase;margin-bottom:.5rem;font-weight:600">
    Portfolio Filters
    </div>""", unsafe_allow_html=True)

    states = sorted(df["state"].dropna().unique())
    s_sel  = st.sidebar.multiselect("State", states, placeholder="All States")
    sizes  = sorted(df["size_bucket"].dropna().unique())
    sz_sel = st.sidebar.multiselect("Bank Size", sizes, placeholder="All Sizes")
    r_sel  = st.sidebar.multiselect("Risk Tier", ["HIGH","MEDIUM","LOW"],
                                     default=["HIGH","MEDIUM","LOW"])

    st.sidebar.markdown("""
    <div style="height:1px;background:rgba(70,149,200,0.3);margin:.9rem 0"></div>
    """, unsafe_allow_html=True)

    m = load_metrics()
    if m and "Random Forest" in m:
        rf = m["Random Forest"]
        st.sidebar.markdown("""
        <div style="font-size:.65rem;letter-spacing:.12em;color:#4695C8;
             text-transform:uppercase;margin-bottom:.5rem;font-weight:600">
        ML Model · Random Forest
        </div>""", unsafe_allow_html=True)
        c1, c2 = st.sidebar.columns(2)
        c1.metric("ROC-AUC",  f"{rf['roc_auc']:.3f}")
        c2.metric("Accuracy", f"{rf['accuracy']:.1%}")
        c1.metric("F1",       f"{rf['f1']:.3f}")
        c2.metric("Recall",   f"{rf['recall']:.3f}")

    st.sidebar.markdown("""
    <div style="height:1px;background:rgba(70,149,200,0.3);margin:.9rem 0"></div>
    <div style="font-size:.65rem;letter-spacing:.12em;color:#4695C8;text-transform:uppercase;margin-bottom:.6rem;font-weight:600">Risk Classification</div>
      <span style="color:#E03030">●</span> High Risk<br>
      <span style="color:#F0C020">●</span> Medium Risk<br>
      <span style="color:#28B060">●</span> Low Risk<br>
    </div>
    """, unsafe_allow_html=True)
    return s_sel, sz_sel, r_sel

def apply_filters(df, s, sz, r):
    if s:  df = df[df["state"].isin(s)]
    if sz: df = df[df["size_bucket"].isin(sz)]
    if r:  df = df[df["risk_tier"].isin(r)]
    return df

# ── KRI Cards ─────────────────────────────────────────────────────────────────
def kri_cards(df):
    n    = len(df)
    high = (df["risk_tier"]=="HIGH").sum()
    car  = df["capital_adequacy_ratio"].median()
    npl  = df["npl_ratio"].median()
    roa  = df["ROA"].median()
    ldr  = df["loan_to_deposit_ratio"].median()

    def card(label, val, sub, pct, color):
        return f"""<div class="kri-card">
          <div class="kc-label">{label}</div>
          <div class="kc-val">{val}</div>
          <div class="kc-sub">{sub}</div>
          <div class="kc-bar"><div class="kc-fill" style="width:{min(pct,100):.0f}%;background:{color}"></div></div>
        </div>"""

    cols = st.columns(6)
    items = [
        ("Total Institutions",  f"{n:,}",       "Active FDIC-insured",         100,             "#187ABA"),
        ("High Risk ⚠",         f"{high:,}",    f"{high/n*100:.1f}% of book",  high/n*100,      "#E03030"),
        ("Capital Ratio",       f"{car:.1f}%",  "Median · Basel III min 8%",   min(car/20*100,100), "#B5965A"),
        ("NPL Ratio",           f"{npl:.2f}%",  "Median · Threshold 5%",       min(npl/10*100,100), "#C8382A" if npl>2 else "#00A8A8"),
        ("Return on Assets",    f"{roa:.2f}%",  "Median · Healthy >1%",        min(roa/3*100,100),  "#00A8A8" if roa>1 else "#B5965A"),
        ("Loan / Deposit",      f"{ldr:.1f}%",  "Median · Threshold 90%",      min(ldr,100),        "#187ABA"),
    ]
    for col,(lbl,val,sub,pct,clr) in zip(cols,items):
        col.markdown(card(lbl,val,sub,pct,clr), unsafe_allow_html=True)

# ═══════════════ TAB 1: EXECUTIVE SUMMARY ════════════════════════════════════
def tab_executive(df, failures):
    n    = len(df)
    high = (df["risk_tier"]=="HIGH").sum()
    med  = (df["risk_tier"]=="MEDIUM").sum()
    low  = (df["risk_tier"]=="LOW").sum()

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-title">Risk Tier Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=["High Risk","Medium Risk","Low Risk"],
            values=[high, med, low], hole=.62,
            marker=dict(colors=["#C8382A","#B5965A","#00A8A8"],
                        line=dict(color="#002B51", width=2)),
            textfont=dict(family="IBM Plex Sans", color="#FFFFFF", size=11),
            hovertemplate="<b>%{label}</b><br>%{value:,} banks · %{percent}<extra></extra>"
        ))
        fig.add_annotation(text=f"<b>{n:,}</b>", x=.5, y=.57,
            font=dict(family="Libre Baskerville", size=24, color="#FFFFFF"), showarrow=False)
        fig.add_annotation(text="TOTAL", x=.5, y=.43,
            font=dict(family="IBM Plex Sans", size=9, color="#4695C8"), showarrow=False)
        fig.update_layout(**pl(height=295,
            legend=dict(orientation="h", y=-.06, x=.5, xanchor="center",
                        font=dict(color="#A3CAE3", size=10), bgcolor="rgba(0,0,0,0)")))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Capital Ratio Distribution</div>', unsafe_allow_html=True)
        car = df["capital_adequacy_ratio"].dropna()
        fig = go.Figure(go.Histogram(
            x=car, nbinsx=40,
            marker=dict(color="#B5965A", opacity=.85, line=dict(color="#002B51", width=.4)),
            hovertemplate="CAR %{x:.1f}%: %{y} banks<extra></extra>"
        ))
        fig.add_vline(x=8,  line=dict(color="#E04040", dash="dash", width=1.5),
                      annotation=dict(text="Min 8%", font=dict(color="#C8382A", size=9), y=.92))
        fig.add_vline(x=10, line=dict(color="#F0A030", dash="dash", width=1.5),
                      annotation=dict(text="Well-Cap", font=dict(color="#E07820", size=9), y=.78))
        fig.update_layout(**pl(height=295,
            xaxis_title="Capital Adequacy Ratio (%)", yaxis_title="Institutions"))
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.markdown('<div class="section-title">Portfolio by Bank Size</div>', unsafe_allow_html=True)
        order = ["Systemically Important (>$250B)","Large ($10B-$250B)",
                 "Mid-Size ($1B-$10B)","Community ($100M-$1B)","Small (<$100M)"]
        sz = df["size_bucket"].value_counts().reindex(order).dropna().reset_index()
        sz.columns = ["Size","Count"]
        labels = {"Systemically Important (>$250B)":"Systemic >$250B",
                  "Large ($10B-$250B)":"Large $10-250B",
                  "Mid-Size ($1B-$10B)":"Mid $1-10B",
                  "Community ($100M-$1B)":"Community",
                  "Small (<$100M)":"Small"}
        sz["Label"] = sz["Size"].map(labels)
        fig = go.Figure(go.Bar(
            x=sz["Count"], y=sz["Label"], orientation="h",
            marker=dict(color=["#D4AF6A","#B5965A","#187ABA","#003A6B","#002B51"][:len(sz)],
                        line=dict(color="#002B51", width=.4)),
            text=sz["Count"].map("{:,}".format), textposition="outside",
            textfont=dict(color="#A3CAE3", size=10),
            hovertemplate="<b>%{y}</b>: %{x:,}<extra></extra>"
        ))
        fig.update_layout(**pl(height=295, xaxis_title="Institutions"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="ms-line"></div>', unsafe_allow_html=True)
    c4, c5 = st.columns([3,2])

    with c4:
        st.markdown('<div class="section-title">Capital Adequacy vs Return on Assets</div>', unsafe_allow_html=True)
        samp = df.sample(min(800,n), random_state=42).copy()
        samp["_sz"] = samp["npl_ratio"].fillna(0).clip(0,15)
        fig = px.scatter(samp, x="capital_adequacy_ratio", y="ROA",
                         color="risk_tier", size="_sz",
                         color_discrete_map=MS_RISK,
                         hover_data=["city","state","size_bucket"],
                         opacity=.75, size_max=16,
                         labels={"capital_adequacy_ratio":"Capital Adequacy Ratio (%)","ROA":"ROA (%)"})
        fig.add_vline(x=8, line=dict(color="#E04040", dash="dot", width=1.2))
        fig.add_hline(y=0, line=dict(color="#F0A030", dash="dot", width=1.2))
        fig.update_layout(**pl(height=320,
            legend=dict(title="Risk Tier", font=dict(color="#A3CAE3", size=10),
                        bgcolor="rgba(0,14,35,0.6)", bordercolor="rgba(70,149,200,0.2)",
                        borderwidth=1)))
        st.plotly_chart(fig, use_container_width=True)

    with c5:
        st.markdown('<div class="section-title">Bank Failures by Decade</div>', unsafe_allow_html=True)
        if "fail_decade" in failures.columns and "insurance_cost_millions" in failures.columns:
            dec = failures.groupby("fail_decade").agg(
                count=("CERT","count"), cost=("insurance_cost_millions","sum")
            ).reset_index().dropna()
            dec = dec[dec["fail_decade"]>=1980]
            fig = make_subplots(specs=[[{"secondary_y":True}]])
            fig.add_trace(go.Bar(x=dec["fail_decade"], y=dec["count"],
                name="Failures", marker_color="#B5965A", opacity=.85,
                hovertemplate="%{x}s: %{y} failures<extra></extra>"), secondary_y=False)
            fig.add_trace(go.Scatter(x=dec["fail_decade"], y=dec["cost"],
                name="Cost $M", line=dict(color="#E04040", width=2.5),
                mode="lines+markers", marker=dict(size=5, color="#E04040"),
                hovertemplate="%{x}s: $%{y:.0f}M<extra></extra>"), secondary_y=True)
            fig.update_layout(**pl(height=320,
                legend=dict(orientation="h", y=-.12, font=dict(color="#A3CAE3", size=10),
                            bgcolor="rgba(0,0,0,0)")))
            fig.update_yaxes(title_text="# Failures", secondary_y=False,
                             gridcolor="rgba(70,149,200,0.1)", tickfont=dict(color="#A3CAE3"))
            fig.update_yaxes(title_text="Cost ($M)", secondary_y=True,
                             gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#A3CAE3"))
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════ TAB 2: RISK HEATMAP ═════════════════════════════════════════
def tab_heatmap(df):
    st.caption("Note: Risk Tier filter does not apply to the heatmap — it always shows all tiers per state for accurate geographic risk distribution.")
    df = df.copy()
    df["st2"] = df["state"].str.upper().map(STATE_MAP).fillna(df["state"])
    ks = df.groupby("st2").agg(
        state_name=("state","first"),
        bank_count=("cert_id","count"),
        avg_car=("capital_adequacy_ratio","median"),
        avg_npl=("npl_ratio","median"),
        avg_roa=("ROA","median"),
        high_pct=("risk_tier", lambda x:(x=="HIGH").mean()*100)
    ).reset_index()

    col_s, _ = st.columns([2,3])
    with col_s:
        metric = st.selectbox("KRI Metric to Display",
            ["high_pct","avg_car","avg_npl","avg_roa"],
            format_func=lambda x:{"high_pct":"% High Risk Banks",
                "avg_car":"Median Capital Ratio","avg_npl":"Median NPL Ratio","avg_roa":"Median ROA"}[x])

    # Traffic light thresholds per metric
    THRESHOLDS = {
        "high_pct": {"green": 5,  "yellow": 15},   # % high risk banks
        "avg_car":  {"red": 8,    "yellow": 10},    # capital ratio (higher = better, reversed)
        "avg_npl":  {"green": 2,  "yellow": 5},     # NPL ratio
        "avg_roa":  {"red": 0,    "yellow": 0.5},   # ROA (higher = better, reversed)
    }

    def traffic_color(val, metric):
        t = THRESHOLDS.get(metric, {})
        if metric in ["avg_car", "avg_roa"]:  # higher is better
            red_t    = t.get("red", 0)
            yellow_t = t.get("yellow", 0.5)
            if val <= red_t:    return "#A01818"
            elif val <= yellow_t: return "#C8A820"
            else:               return "#1A6640"
        else:  # lower is better
            green_t  = t.get("green", 5)
            yellow_t = t.get("yellow", 15)
            if val <= green_t:    return "#1A6640"
            elif val <= yellow_t: return "#C8A820"
            else:                 return "#A01818"

    ks["traffic_color"] = ks[metric].apply(lambda v: traffic_color(v, metric))
    ks["risk_label"] = ks[metric].apply(lambda v: (
        "🟢 Low Risk" if traffic_color(v, metric) == "#28B060"
        else "🟡 Medium Risk" if traffic_color(v, metric) == "#F0C020"
        else "🔴 High Risk"
    ))

    # Intuitive colorscales per metric
    METRIC_COLORSCALE = {
        # % High Risk Banks: low=safe green, high=danger red
        "high_pct": [[0,"#1A5C30"],[0.25,"#2E8B45"],[0.5,"#D4A020"],[0.75,"#C85020"],[1,"#8B1010"]],
        # NPL Ratio: low=healthy green, high=distressed red
        "avg_npl":  [[0,"#1A5C30"],[0.25,"#2E8B45"],[0.5,"#D4A020"],[0.75,"#C85020"],[1,"#8B1010"]],
        # Capital Ratio: low=undercapitalized red, high=well-capitalized blue (MS brand)
        "avg_car":  [[0,"#8B1010"],[0.25,"#C85020"],[0.5,"#D4A020"],[0.75,"#1868A0"],[1,"#0A3D6B"]],
        # ROA: negative/low=red, positive/high=blue-green
        "avg_roa":  [[0,"#8B1010"],[0.3,"#C85020"],[0.5,"#D4A020"],[0.7,"#1A7A50"],[1,"#0A4530"]],
    }
    chosen_scale = METRIC_COLORSCALE.get(metric, [[0,"#1A5C30"],[0.5,"#D4A020"],[1,"#8B1010"]])

    fig = go.Figure(go.Choropleth(
        locations=ks["st2"],
        locationmode="USA-states",
        z=ks[metric],
        colorscale=chosen_scale,
        marker_line_color="rgba(70,149,200,0.4)",
        marker_line_width=1,
        colorbar=dict(
            tickfont=dict(color="#A3CAE3", size=10),
            title=dict(text=metric.replace("_"," ").title(), font=dict(color="#A3CAE3")),
            bgcolor="rgba(0,43,81,0.6)",
            bordercolor="rgba(70,149,200,0.3)",
            borderwidth=1,
        ),
        customdata=np.stack([ks["state_name"], ks["bank_count"],
                             ks["avg_car"], ks["avg_npl"],
                             ks["avg_roa"], ks["risk_label"]], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Risk Status: %{customdata[5]}<br>"
            "Banks: %{customdata[1]:,}<br>"
            "Capital Ratio: %{customdata[2]:.1f}%<br>"
            "NPL Ratio: %{customdata[3]:.2f}%<br>"
            "ROA: %{customdata[4]:.2f}%<extra></extra>"
        )
    ))
    fig.update_layout(**pl(height=500, geo=dict(
        scope="usa",
        bgcolor="rgba(0,0,0,0)",
        lakecolor="rgba(0,43,81,0.5)",
        landcolor="#003A6B",
        subunitcolor="rgba(70,149,200,0.4)",
        showlakes=True,
    )))

    # Subtle traffic light legend
    t = THRESHOLDS.get(metric, {})
    if metric in ["high_pct", "avg_npl"]:
        g_thresh = t.get("green", 5)
        y_thresh = t.get("yellow", 15)
        legend_items = [
            ("#1A9A50", "rgba(26,154,80,0.1)",  "Low Risk",    f"< {g_thresh}%"),
            ("#C8A820", "rgba(200,168,32,0.1)",  "Medium Risk", f"{g_thresh}–{y_thresh}%"),
            ("#C83030", "rgba(200,48,48,0.1)",   "High Risk",   f"> {y_thresh}%"),
        ]
    else:
        r_thresh = t.get("red", 0)
        y_thresh = t.get("yellow", 0.5)
        legend_items = [
            ("#1A9A50", "rgba(26,154,80,0.1)",   "Low Risk",    f"> {y_thresh}%"),
            ("#C8A820", "rgba(200,168,32,0.1)",  "Medium Risk", f"{r_thresh}–{y_thresh}%"),
            ("#C83030", "rgba(200,48,48,0.1)",   "High Risk",   f"< {r_thresh}%"),
        ]
    legend_html = "<div style='display:flex;gap:12px;margin-bottom:12px'>"
    for clr, bg, label, rng in legend_items:
        legend_html += f"""
        <div style='display:flex;align-items:center;gap:7px;padding:.35rem .9rem;
             background:{bg};border:1px solid {clr}40;border-radius:6px'>
          <div style='width:10px;height:10px;border-radius:50%;background:{clr};flex-shrink:0'></div>
          <span style='font-size:.75rem;color:#D1E4F1;font-weight:500'>{label}</span>
          <span style='font-size:.72rem;color:#8AACCC'>{rng}</span>
        </div>"""
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="ms-line"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Top 15 States · High-Risk Concentration</div>', unsafe_allow_html=True)
        top = ks.nlargest(15,"high_pct")
        fig2 = go.Figure(go.Bar(
            x=top["high_pct"], y=top["st2"], orientation="h",
            marker=dict(color=top["high_pct"],
                        colorscale=[[0,"#003A6B"],[0.4,"#187ABA"],[0.7,"#B5965A"],[1,"#C8382A"]],
                        line=dict(color="#002B51", width=.4)),
            text=top["high_pct"].map("{:.1f}%".format),
            textposition="outside", textfont=dict(color="#A3CAE3", size=9),
            hovertemplate="<b>%{y}</b>: %{x:.1f}% high-risk<extra></extra>"
        ))
        fig2.update_layout(**pl(height=380, xaxis_title="% High Risk Banks"))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">State KRI Summary</div>', unsafe_allow_html=True)
        d = ks[["st2","bank_count","avg_car","avg_npl","avg_roa","high_pct"]].copy()
        d.columns = ["State","Banks","Cap Ratio","NPL","ROA","% High Risk"]
        d = d.sort_values("% High Risk", ascending=False).head(20)
        for col in ["Cap Ratio","NPL","ROA","% High Risk"]:
            d[col] = d[col].map("{:.2f}%".format)
        st.dataframe(d, use_container_width=True, height=380, hide_index=True)

# ═══════════════ TAB 3: MACRO & INTEREST RATES ═══════════════════════════════
def tab_macro(df, rates_df, events_df):
    if rates_df.empty:
        st.warning("Interest rate data not found. Run `python src/pipeline/fred_pipeline.py` first.")
        st.info("The pipeline will fetch Fed Funds Rate, 10Y Treasury, and 2Y Treasury from FRED.")
        return

    st.caption("Federal Reserve interest rate data · FRED H.15 Release · Impact on banking sector risk")

    # ── Row 1: KRI summary cards ──
    latest = rates_df.iloc[-1]
    prev   = rates_df.iloc[-2] if len(rates_df) > 1 else latest

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Fed Funds Rate",     f"{latest['fed_funds']:.2f}%",
              f"{latest['fed_funds']-prev['fed_funds']:+.2f}% QoQ")
    c2.metric("10Y Treasury",       f"{latest['treasury_10y']:.2f}%",
              f"{latest['treasury_10y']-prev['treasury_10y']:+.2f}% QoQ")
    c3.metric("2Y Treasury",        f"{latest['treasury_2y']:.2f}%",
              f"{latest['treasury_2y']-prev['treasury_2y']:+.2f}% QoQ")
    spread = latest["yield_curve_spread"]
    c4.metric("Yield Curve (2s10s)",f"{spread:.2f}%",
              "⚠️ Inverted" if spread < 0 else "Normal",
              delta_color="inverse" if spread < 0 else "normal")

    st.markdown('<div class="ms-line"></div>', unsafe_allow_html=True)

    # ── Row 2: Rate history + yield curve ──
    c1, c2 = st.columns([3,2])

    with c1:
        st.markdown('<div class="section-title">Fed Funds Rate & Treasury Yields (2019–2025)</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rates_df["date"], y=rates_df["fed_funds"],
            name="Fed Funds Rate", line=dict(color="#187ABA", width=2.5),
            hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>Fed Funds</extra>"))
        fig.add_trace(go.Scatter(x=rates_df["date"], y=rates_df["treasury_10y"],
            name="10Y Treasury", line=dict(color="#E8C060", width=2),
            hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>10Y</extra>"))
        fig.add_trace(go.Scatter(x=rates_df["date"], y=rates_df["treasury_2y"],
            name="2Y Treasury", line=dict(color="#28B060", width=2, dash="dot"),
            hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>2Y</extra>"))

        # Add macro event annotations
        if not events_df.empty:
            for _, ev in events_df.iterrows():
                try:
                    fig.add_vline(x=pd.to_datetime(ev["date"]),
                        line=dict(color="rgba(200,56,42,0.4)", dash="dot", width=1.2),
                        annotation=dict(text=ev["event"], font=dict(color="#C8382A", size=8),
                                        textangle=-90, y=0.98, yref="paper"))
                except: pass

        fig.add_hrect(y0=5.0, y1=5.5, fillcolor="rgba(200,56,42,0.06)",
                      line_width=0, annotation_text="Rate Peak Zone",
                      annotation_font=dict(color="#C8382A", size=9))
        fig.update_layout(**pl(height=360,
            legend=dict(orientation="h", y=-.1, font=dict(color="#A3CAE3", size=10),
                        bgcolor="rgba(0,0,0,0)")))
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Yield Curve Spread (10Y − 2Y)</div>', unsafe_allow_html=True)
        spread_df = rates_df.dropna(subset=["yield_curve_spread"])
        colors = ["#E03030" if v < 0 else "#28B060" for v in spread_df["yield_curve_spread"]]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=spread_df["date"], y=spread_df["yield_curve_spread"],
            marker_color=colors, marker_line_width=0,
            hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra></extra>"
        ))
        fig2.add_hline(y=0, line=dict(color="#A3CAE3", width=1, dash="dash"))
        fig2.add_hrect(y0=-2, y1=0, fillcolor="rgba(200,48,48,0.08)", line_width=0,
                       annotation_text="Inverted (Recession Risk)",
                       annotation_font=dict(color="#C8382A", size=8))
        fig2.update_layout(**pl(height=360))
        fig2.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="ms-line"></div>', unsafe_allow_html=True)

    # ── Row 3: Rate impact on banking sector ──
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-title">Rate Environment vs Bank Failures</div>', unsafe_allow_html=True)
        # Map failure years to rate environment
        fail_yr = None
        try:
            conn = sqlite3.connect(DB_PATH)
            fail_yr = pd.read_sql(
                "SELECT fail_year, COUNT(*) as failures FROM failures "
                "WHERE fail_year >= 2019 GROUP BY fail_year", conn)
            conn.close()
        except: pass

        if fail_yr is not None and not fail_yr.empty:
            # Merge with annual avg rate
            rates_df["year"] = rates_df["date"].dt.year
            annual_rates = rates_df.groupby("year")["fed_funds"].mean().reset_index()
            annual_rates.columns = ["fail_year","avg_fed_funds"]
            merged = fail_yr.merge(annual_rates, on="fail_year", how="left")

            fig3 = make_subplots(specs=[[{"secondary_y":True}]])
            fig3.add_trace(go.Bar(x=merged["fail_year"], y=merged["failures"],
                name="Bank Failures", marker_color="#187ABA", opacity=.85,
                hovertemplate="%{x}: %{y} failures<extra></extra>"), secondary_y=False)
            fig3.add_trace(go.Scatter(x=merged["fail_year"], y=merged["avg_fed_funds"],
                name="Fed Funds Rate", line=dict(color="#E8C060", width=2.5),
                mode="lines+markers", marker=dict(size=6),
                hovertemplate="%{x}: %{y:.2f}%<extra>Fed Funds</extra>"), secondary_y=True)
            fig3.update_layout(**pl(height=300,
                legend=dict(orientation="h", y=-.15, font=dict(color="#A3CAE3",size=10),
                            bgcolor="rgba(0,0,0,0)")))
            fig3.update_yaxes(title_text="# Failures", secondary_y=False,
                              gridcolor="rgba(70,149,200,0.1)", tickfont=dict(color="#A3CAE3"))
            fig3.update_yaxes(title_text="Fed Funds Rate %", secondary_y=True,
                              gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#A3CAE3"),
                              ticksuffix="%")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No recent failure data to overlay.")

    with c4:
        st.markdown('<div class="section-title">Rate Environment Impact on Portfolio KRIs</div>', unsafe_allow_html=True)

        # Show how current rate environment affects risk metrics
        rate_env = latest.get("rate_environment", "Stable")
        fed_rate = float(latest["fed_funds"])
        spread_v = float(latest["yield_curve_spread"])

        st.markdown(f"""
        <div style="background:rgba(0,43,81,0.5);border:1px solid rgba(70,149,200,0.2);
             border-radius:10px;padding:1.2rem;margin-bottom:.8rem">
          <div style="font-size:.65rem;letter-spacing:.12em;color:#4695C8;
               text-transform:uppercase;margin-bottom:.6rem;font-weight:600">
            Current Rate Environment
          </div>
          <div style="font-family:'Libre Baskerville',serif;font-size:1.6rem;
               font-weight:700;color:#E8C060;margin-bottom:.4rem">{rate_env}</div>
          <div style="font-size:.8rem;color:#A3CAE3;line-height:1.8">
            Fed Funds: <b style="color:#fff">{fed_rate:.2f}%</b><br>
            Yield Curve: <b style="color:{'#E03030' if spread_v < 0 else '#28B060'}">
              {'⚠️ Inverted ' if spread_v < 0 else '✅ Normal '}{spread_v:+.2f}%</b><br>
            Risk Signal: <b style="color:#fff">
              {'High — inverted curve historically precedes recessions' if spread_v < 0
               else 'Moderate — elevated rates pressuring NIM' if fed_rate > 4
               else 'Low — accommodative rate environment'}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # NIM analysis — how rate changes affect bank profitability
        avg_nim = df["net_interest_margin"].median() if "net_interest_margin" in df.columns else None
        st.markdown(f"""
        <div style="background:rgba(0,43,81,0.5);border:1px solid rgba(70,149,200,0.2);
             border-radius:10px;padding:1.2rem">
          <div style="font-size:.65rem;letter-spacing:.12em;color:#4695C8;
               text-transform:uppercase;margin-bottom:.6rem;font-weight:600">
            NIM vs Rate Environment
          </div>
          <div style="font-size:.8rem;color:#A3CAE3;line-height:1.9">
            Portfolio Median NIM: <b style="color:#fff">{f'{avg_nim:.2f}%' if avg_nim else 'N/A'}</b><br>
            Fed Funds Rate: <b style="color:#fff">{fed_rate:.2f}%</b><br>
            NIM Compression Risk: <b style="color:{'#E03030' if avg_nim and avg_nim < fed_rate/2 else '#28B060'}">
              {'High — NIM below half of Fed rate' if avg_nim and avg_nim < fed_rate/2
               else 'Low — NIM healthy relative to rates'}</b><br>
            Rate Sensitivity: Banks with long-duration assets most exposed to<br>
            mark-to-market losses if rates remain elevated.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="ms-line"></div>', unsafe_allow_html=True)

    # ── Rate history table ──
    st.markdown('<div class="section-title">Historical Rate Data (Quarterly)</div>', unsafe_allow_html=True)
    display = rates_df[["date","fed_funds","treasury_10y","treasury_2y",
                         "yield_curve_spread","rate_environment"]].copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    for col in ["fed_funds","treasury_10y","treasury_2y","yield_curve_spread"]:
        display[col] = display[col].map("{:.2f}%".format)
    display.columns = ["Date","Fed Funds","10Y Treasury","2Y Treasury","Yield Curve Spread","Environment"]
    display = display.iloc[::-1].reset_index(drop=True)
    st.dataframe(display, use_container_width=True, height=300, hide_index=True)


# ═══════════════ TAB 3: ML SCORING ═══════════════════════════════════════════
def tab_ml(df):
    metrics = load_metrics()
    fi      = load_fi()

    if metrics:
        cols = st.columns(len(metrics))
        for i,(name,m) in enumerate(metrics.items()):
            cols[i].markdown(f"""
            <div class="kri-card">
              <div class="kc-label">{name}</div>
              <div class="kc-val">{m['roc_auc']:.3f}</div>
              <div class="kc-sub">AUC · Acc {m['accuracy']:.1%} · F1 {m['f1']:.3f}</div>
              <div class="kc-bar">
                <div class="kc-fill" style="width:{m['roc_auc']*100:.0f}%;background:#187ABA"></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    st.markdown('<div class="ms-line"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Model Performance Comparison</div>', unsafe_allow_html=True)
        if metrics:
            rows = []
            for name,m in metrics.items():
                for met,val in m.items():
                    if met in ["accuracy","precision","recall","f1","roc_auc"]:
                        rows.append({"Model":name,"Metric":met.upper().replace("_","-"),"Score":val})
            mdf = pd.DataFrame(rows)
            fig = px.bar(mdf, x="Metric", y="Score", color="Model", barmode="group",
                         color_discrete_sequence=["#187ABA","#B5965A","#00A8A8"])
            fig.update_layout(**pl(height=300, yaxis_range=[0,1],
                legend=dict(font=dict(color="#A3CAE3",size=10), bgcolor="rgba(0,0,0,0)")))
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
        if not fi.empty:
            fig = go.Figure(go.Bar(
                x=fi["importance"], y=fi["feature"], orientation="h",
                marker=dict(color=fi["importance"],
                            colorscale=[[0,"#003A6B"],[.4,"#187ABA"],[.75,"#B5965A"],[1,"#D4AF6A"]],
                            line=dict(color="#002B51", width=.4)),
                text=fi["importance"].map("{:.3f}".format),
                textposition="outside", textfont=dict(color="#A3CAE3", size=9),
                hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>"
            ))
            fig.update_layout(**pl(height=300, xaxis_title="Importance Score"))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="ms-line"></div>', unsafe_allow_html=True)
    c3, c4 = st.columns([3,2])

    with c3:
        st.markdown('<div class="section-title">High-Risk Institution Watch List</div>', unsafe_allow_html=True)
        if "failure_probability" in df.columns:
            avail = [c for c in ["bank_name","city","state","size_bucket","ROA",
                                  "capital_adequacy_ratio","npl_ratio",
                                  "failure_probability","total_flags","risk_cluster_label"]
                     if c in df.columns]
            watch = df[df["failure_probability"]>.35].sort_values(
                "failure_probability", ascending=False)[avail].head(25).copy()
            if "failure_probability" in watch.columns:
                watch["failure_probability"] = watch["failure_probability"].map("{:.1%}".format)
            st.dataframe(watch, use_container_width=True, height=340, hide_index=True)
        else:
            st.info("Run `python src/ml/risk_engine.py` to generate ML scores.")

    with c4:
        st.markdown('<div class="section-title">K-Means Risk Clusters</div>', unsafe_allow_html=True)
        if "risk_cluster_label" in df.columns:
            cd = df["risk_cluster_label"].value_counts().reset_index()
            cd.columns = ["Cluster","Count"]
            clr_map = {"Low Risk":"#00A8A8","Elevated Risk":"#B5965A",
                       "High Risk":"#C8382A","Critical Risk":"#6A1A10"}
            fig = px.bar(cd, x="Cluster", y="Count", color="Cluster",
                         color_discrete_map=clr_map, text="Count")
            fig.update_traces(textposition="outside", textfont=dict(color="#A3CAE3",size=10))
            fig.update_layout(**pl(height=340, showlegend=False))
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════ TAB 4: CONTROL TESTING ══════════════════════════════════════
def tab_control(df, failures):
    n = len(df)
    controls = [
        ("Capital < 8% (Critical)",  "Capital",      (df["capital_adequacy_ratio"]<8).sum(),  "#E04040"),
        ("Capital < 10% (Warning)",  "Capital",      (df["capital_adequacy_ratio"]<10).sum(), "#F0A030"),
        ("NPL > 5% (Warning)",       "Credit",       (df["npl_ratio"]>5).sum(),               "#F0A030"),
        ("NPL > 10% (Critical)",     "Credit",       (df["npl_ratio"]>10).sum(),              "#E04040"),
        ("Negative ROA",             "Profitability",(df["ROA"]<0).sum(),                     "#C8992A"),
        ("Loan/Deposit > 90%",       "Liquidity",    (df["loan_to_deposit_ratio"]>90).sum(),  "#187ABA"),
        ("Cost/Income > 70%",        "Efficiency",   (df["cost_to_income_ratio"]>70).sum(),   "#10C090"),
    ]
    cd = pd.DataFrame(controls, columns=["Control","Type","Exceptions","Color"])
    cd["Rate"]   = (cd["Exceptions"]/n*100).round(2)
    cd["Status"] = cd["Rate"].apply(lambda x:"🔴 Critical" if x>20 else("🟡 Elevated" if x>5 else "🟢 Normal"))

    c1, c2 = st.columns([1.6,1])
    with c1:
        st.markdown('<div class="section-title">Control Exception Analysis</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for _, row in cd.iterrows():
            fig.add_trace(go.Bar(
                x=[row["Exceptions"]], y=[row["Control"]], orientation="h",
                marker_color=row["Color"], showlegend=False,
                text=[f"{row['Rate']:.1f}%"], textposition="outside",
                textfont=dict(color="#A3CAE3", size=9),
                hovertemplate=f"<b>{row['Control']}</b><br>{row['Exceptions']:,} ({row['Rate']:.1f}%)<extra></extra>"
            ))
        fig.update_layout(**pl(height=360, xaxis_title="Number of Exceptions"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Control Status</div>', unsafe_allow_html=True)
        disp = cd[["Control","Exceptions","Rate","Status"]].copy()
        disp.columns = ["Control","Exceptions","Rate (%)","Status"]
        st.dataframe(disp, use_container_width=True, height=360, hide_index=True)

    st.markdown('<div class="ms-line"></div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-title">Historical Failure Analysis by Decade</div>', unsafe_allow_html=True)
        if "fail_decade" in failures.columns:
            dec = failures.groupby("fail_decade").agg(
                failures=("CERT","count"), cost=("insurance_cost_millions","sum")
            ).reset_index().dropna()
            dec["avg_cost"] = (dec["cost"]/dec["failures"]).round(1)
            fig = px.bar(dec, x="fail_decade", y="failures",
                         color="avg_cost",
                         color_continuous_scale=[[0,"#003A6B"],[0.5,"#B5965A"],[1,"#C8382A"]],
                         text="failures",
                         labels={"fail_decade":"Decade","failures":"Failures","avg_cost":"Avg Cost $M"})
            fig.update_traces(textposition="outside", textfont=dict(color="#A3CAE3",size=9))
            fig.update_layout(**pl(height=300))
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">Risk-Based Sampling (5% of High-Risk)</div>', unsafe_allow_html=True)
        hr = df[df["risk_tier"]=="HIGH"]
        sn = max(1, int(len(hr)*.05))
        samp = hr.sample(min(sn,len(hr)), random_state=42)
        avail = [c for c in ["bank_name","city","state","capital_adequacy_ratio","ROA","npl_ratio"]
                 if c in samp.columns]
        st.caption(f"Random 5% of {len(hr)} high-risk institutions = {sn} selected for review")
        st.dataframe(samp[avail].reset_index(drop=True), use_container_width=True, height=270, hide_index=True)

# ═══════════════ TAB 5: INSTITUTION LOOKUP ════════════════════════════════════
def tab_search(df):
    st.caption("Search by bank name, city or state · Live query against FDIC database")

    if "sr" not in st.session_state:
        st.session_state.sr = None

    def _search():
        q = st.session_state["_q"].strip()
        if not q:
            st.session_state.sr = None
            return
        try:
            conn = sqlite3.connect(DB_PATH)
            cols = [r[1] for r in conn.execute("PRAGMA table_info(institutions)").fetchall()]
            nc   = next((c for c in ["bank_name","INSTNAME"] if c in cols), None)
            res  = pd.DataFrame()
            if nc:
                res = pd.read_sql(
                    f"SELECT * FROM institutions WHERE UPPER({nc}) LIKE UPPER(?) LIMIT 20",
                    conn, params=(f"%{q}%",))
            if res.empty:
                res = pd.read_sql(
                    "SELECT * FROM institutions WHERE UPPER(city) LIKE UPPER(?) OR UPPER(state) LIKE UPPER(?) LIMIT 20",
                    conn, params=(f"%{q}%",f"%{q}%"))
            conn.close()
            # Filter out institutions with missing/bad data
            if not res.empty:
                asset_col = "ASSET" if "ASSET" in res.columns else None
                if asset_col:
                    res[asset_col] = pd.to_numeric(res[asset_col], errors="coerce")
                    # Remove institutions with assets < $1M (thousands) = $1B
                    # Keep only institutions with at least $10M in assets (10000 in thousands)
                    res = res[res[asset_col].fillna(0) > 10000]
                # Remove rows where both capital_ratio AND npl_ratio are NaN
                if "capital_adequacy_ratio" in res.columns and "npl_ratio" in res.columns:
                    res["capital_adequacy_ratio"] = pd.to_numeric(res["capital_adequacy_ratio"], errors="coerce")
                    res["npl_ratio"] = pd.to_numeric(res["npl_ratio"], errors="coerce")
                    res = res[~(res["capital_adequacy_ratio"].isna() & res["npl_ratio"].isna())]
            st.session_state.sr = res
        except Exception as e:
            st.error(str(e))
            st.session_state.sr = pd.DataFrame()

    ci, cb = st.columns([5,1])
    with ci:
        st.text_input("Institution Search",
                      placeholder="Search bank name, city or state…",
                      key="_q", on_change=_search,
                      label_visibility="collapsed")
    with cb:
        if st.button("Search", use_container_width=True):
            _search()

    res = st.session_state.sr
    if res is None:
        st.markdown("""
        <div style="text-align:center;padding:3rem 0;color:#4695C8">
          <div style="font-size:2.5rem;margin-bottom:.7rem">🏦</div>
          <div style="font-family:'Libre Baskerville',serif;font-size:1.1rem;
               color:#FFFFFF;margin-bottom:.4rem">Institution Risk Profile Lookup</div>
          <div style="font-size:.82rem;color:#A3CAE3">
            Try searching by bank name, state, or city</div>
        </div>""", unsafe_allow_html=True)
        return

    if res.empty:
        st.warning("No institutions found. Try a different search term.")
        return

    st.success(f"**{len(res)}** institutions found")
    for _, row in res.iterrows():
        tier  = str(row.get("risk_tier",""))
        badge = f'<span class="badge badge-{tier.lower()}">{tier}</span>' if tier in ["HIGH","MEDIUM","LOW"] else ""
        nc    = next((c for c in ["bank_name","INSTNAME","city"] if c in row.index), None)
        label = row.get(nc,"Unknown") if nc else "Unknown"

        with st.expander(f"🏦  {label}   —   {row.get('city','')}, {row.get('state','')}"):
            st.markdown(f"**Risk Classification:** {badge}", unsafe_allow_html=True)
            st.markdown('<div class="ms-line" style="margin:.5rem 0"></div>', unsafe_allow_html=True)
            ca,cb2,cc,cd2 = st.columns(4)
            ca.metric("Capital Ratio", fmt(row.get("capital_adequacy_ratio"),1))
            cb2.metric("NPL Ratio",    fmt(row.get("npl_ratio"),2))
            cc.metric("ROA",           fmt(row.get("ROA"),2))
            cd2.metric("Loan/Deposit", fmt(row.get("loan_to_deposit_ratio"),1))
            ce,cf,cg,ch = st.columns(4)
            ce.metric("Risk Tier",   tier or "N/A")
            cf.metric("Size",        row.get("size_bucket","N/A"))
            cg.metric("Total Assets",f'${float(row.get("ASSET",0))/1e9:.2f}B' if pd.notna(row.get("ASSET")) else "N/A")
            ch.metric("Net Income",  f'${float(row.get("NETINC",0))/1e6:.1f}M' if pd.notna(row.get("NETINC")) else "N/A")
            if "failure_probability" in row.index and pd.notna(row.get("failure_probability")):
                prob = float(row["failure_probability"])
                clr  = "#E03030" if prob>.5 else "#F0C020" if prob>.25 else "#28B060"
                st.markdown(f"""
                <div style="margin-top:.6rem;padding:.8rem 1rem;
                     background:rgba(0,43,81,0.6);border-radius:8px;
                     border-left:3px solid {clr}">
                  <span style="color:#4695C8;font-size:.65rem;letter-spacing:.1em;
                        text-transform:uppercase;font-weight:600">ML Failure Risk Score</span><br>
                  <span style="font-family:'Libre Baskerville',serif;font-size:1.8rem;
                        font-weight:700;color:{clr}">{prob:.1%}</span>
                  <span style="color:#A3CAE3;font-size:.8rem">
                    &nbsp;·&nbsp; {row.get('risk_cluster_label','N/A')}</span>
                </div>""", unsafe_allow_html=True)

# ═══════════════ MAIN ════════════════════════════════════════════════════════
def main():
    if not os.path.exists(DB_PATH):
        st.error("⚠️ Database not found. Run the pipeline first:")
        st.code("python src/pipeline/fdic_pipeline.py\npython src/ml/risk_engine.py")
        return

    data         = load_all()
    institutions = to_num(data.get("institutions", pd.DataFrame()))
    failures     = data.get("failures", pd.DataFrame())
    risk_scores  = to_num(data.get("risk_scores",  pd.DataFrame()))

    if not risk_scores.empty and "failure_probability" in risk_scores.columns:
        ml_cols = [c for c in ["failure_probability","risk_score_100","risk_cluster_label",
                                "total_flags","predicted_at_risk"] if c in risk_scores.columns]
        ir = institutions.reset_index(drop=True)
        rr = risk_scores[ml_cols].reset_index(drop=True)
        df = pd.concat([ir, rr], axis=1) if len(ir)==len(rr) else ir
    else:
        df = institutions.copy()

    # ── Header ──
    hc, dc = st.columns([3,1])
    with hc:
        st.markdown('<div class="hero-title">Banking Operational Risk<br>Intelligence Platform</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="hero-meta">FDIC BankFind · Basel III · CCAR · ML Risk Scoring</div>',
                    unsafe_allow_html=True)
    with dc:
        st.markdown("""
        <div style="text-align:right;padding-top:.5rem">
          <div style="font-size:.62rem;letter-spacing:.12em;color:#4695C8;
               text-transform:uppercase;font-weight:600">Reporting Period</div>
          <div style="font-family:'Libre Baskerville',serif;font-size:1.3rem;
               font-weight:700;color:#FFFFFF">Q3 2025</div>
          <div style="font-size:.76rem;color:#A3CAE3">September 30, 2025</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="ms-line"></div>', unsafe_allow_html=True)

    s_sel, sz_sel, r_sel = sidebar(df)
    df_f = apply_filters(df, s_sel, sz_sel, r_sel)

    kri_cards(df_f)
    st.markdown('<div class="ms-line"></div>', unsafe_allow_html=True)

    # Tab nav using buttons + session_state — only reliable way to prevent redirect
    TABS = {
        "📊 Executive":   "exec",
        "🗺️ Heatmap":     "heat",
        "📈 Macro & Rates":"macro",
        "🤖 ML Scoring":  "ml",
        "✅ Control":     "ctrl",
        "🔍 Lookup":      "srch",
    }
    if "tab" not in st.session_state:
        st.session_state.tab = "exec"

    # Render tab buttons
    btn_cols = st.columns(len(TABS))
    for col, (label, key) in zip(btn_cols, TABS.items()):
        active = st.session_state.tab == key
        bg     = "#187ABA" if active else "#003A6B"
        border = "#187ABA" if active else "rgba(70,149,200,0.2)"
        fw     = "700" if active else "400"
        col.markdown(f"""
        <style>
        div[data-testid="stButton"] .tab-{key}{{background:{bg}}}
        </style>""", unsafe_allow_html=True)
        if col.button(label, key=f"tab_{key}", use_container_width=True,
                      type="primary" if active else "secondary"):
            st.session_state.tab = key
            st.rerun()

    st.markdown('<div class="ms-line"></div>', unsafe_allow_html=True)

    tab = st.session_state.tab
    rates_df, events_df = load_rates()
    if tab == "exec":  tab_executive(df_f, failures)
    elif tab == "heat": tab_heatmap(df_f)
    elif tab == "macro": tab_macro(df_f, rates_df, events_df)
    elif tab == "ml":   tab_ml(df_f)
    elif tab == "ctrl": tab_control(df_f, failures)
    elif tab == "srch": tab_search(df_f)

if __name__ == "__main__":
    main()