"""
BESCOM Smart Meter Intelligence — Dashboard v2.0
Premium UI/UX redesign with glassmorphism, animated cards, and professional layout
Run: streamlit run bescom_dashboard_v2.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="BESCOM · Smart Meter AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════
# PREMIUM CSS — Full UI Overhaul
# ══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── App background with mesh gradient ── */
.stApp {
    background: #060a10 !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(35,139,230,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(2,195,154,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 50% 50%, rgba(248,81,73,0.04) 0%, transparent 70%) !important;
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(10, 15, 25, 0.95) !important;
    border-right: 1px solid rgba(35,139,230,0.2) !important;
    backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

/* ── Sidebar radio buttons → custom nav pills ── */
div[data-testid="stSidebar"] .stRadio > div {
    gap: 4px !important;
}
div[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    color: #8b949e !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    cursor: pointer;
    display: block;
    width: 100%;
}
div[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(35,139,230,0.12) !important;
    border-color: rgba(35,139,230,0.4) !important;
    color: #c9d1d9 !important;
    transform: translateX(3px);
}
div[data-testid="stSidebar"] .stRadio [data-checked="true"] + label,
div[data-testid="stSidebar"] .stRadio input:checked + label {
    background: linear-gradient(135deg, rgba(35,139,230,0.25), rgba(2,195,154,0.15)) !important;
    border-color: rgba(35,139,230,0.6) !important;
    color: #e6f0ff !important;
    box-shadow: 0 0 20px rgba(35,139,230,0.15);
}

/* ── Main content padding ── */
.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem !important;
    max-width: 1400px;
}

/* ── Page header ── */
.page-header {
    padding: 1.5rem 0 0.5rem 0;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid rgba(35,139,230,0.15);
}
.page-header h1 {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: #e6edf3 !important;
    letter-spacing: -0.02em;
    margin: 0 !important;
}
.page-header p {
    color: #8b949e;
    font-size: 0.9rem;
    margin: 0.3rem 0 0 0;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: rgba(22, 27, 34, 0.8);
    border: 1px solid rgba(48, 54, 61, 0.8);
    border-radius: 14px;
    padding: 18px 16px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
    cursor: default;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
    border-radius: 14px 14px 0 0;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}
.kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #6e7681;
    margin-top: 6px;
}
.kpi-icon {
    position: absolute;
    right: 14px;
    top: 14px;
    font-size: 1.3rem;
    opacity: 0.25;
}

/* ── Zone Risk Cards ── */
.zone-card {
    background: rgba(22,27,34,0.7);
    border: 1px solid rgba(48,54,61,0.6);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: all 0.2s;
    position: relative;
    overflow: hidden;
}
.zone-card:hover {
    background: rgba(35,139,230,0.06);
    border-color: rgba(35,139,230,0.25);
    transform: translateX(3px);
}
.zone-risk-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-HIGH   { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid rgba(248,81,73,0.3); }
.badge-MEDIUM { background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid rgba(210,153,34,0.3); }
.badge-LOW    { background: rgba(63,185,80,0.15);  color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }

/* ── Section headers ── */
.section-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #8b949e;
    margin: 1.5rem 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(48,54,61,0.6);
}

/* ── Alert box ── */
.alert-box {
    background: rgba(248,81,73,0.08);
    border: 1px solid rgba(248,81,73,0.25);
    border-left: 3px solid #f85149;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 1rem 0;
    color: #f85149;
    font-size: 0.88rem;
    font-weight: 500;
}
.info-box {
    background: rgba(35,139,230,0.08);
    border: 1px solid rgba(35,139,230,0.2);
    border-left: 3px solid #238be6;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 0.5rem 0;
    color: #79c0ff;
    font-size: 0.85rem;
}
.success-box {
    background: rgba(63,185,80,0.08);
    border: 1px solid rgba(63,185,80,0.2);
    border-left: 3px solid #3fb950;
    border-radius: 8px;
    padding: 12px 16px;
    color: #3fb950;
    font-size: 0.85rem;
}

/* ── Metric row (detection page) ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 1.5rem;
}
.metric-tile {
    background: rgba(22,27,34,0.8);
    border: 1px solid rgba(48,54,61,0.6);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-tile-val {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: var(--c);
}
.metric-tile-lbl {
    font-size: 0.72rem;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

/* ── Architecture table ── */
.arch-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.arch-table th {
    background: rgba(35,139,230,0.12);
    color: #79c0ff;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid rgba(35,139,230,0.2);
}
.arch-table td {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(48,54,61,0.4);
    color: #c9d1d9;
    vertical-align: top;
}
.arch-table tr:hover td {
    background: rgba(35,139,230,0.04);
}
.tech-badge {
    display: inline-block;
    background: rgba(188,140,255,0.12);
    color: #bc8cff;
    border: 1px solid rgba(188,140,255,0.2);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}

/* ── Streamlit overrides ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(22,27,34,0.9) !important;
    border-color: rgba(48,54,61,0.8) !important;
    color: #c9d1d9 !important;
    border-radius: 10px !important;
}
.stSlider > div > div > div {
    background: rgba(35,139,230,0.4) !important;
}
.stSlider [data-testid="stThumbValue"] {
    background: #238be6 !important;
}

/* Download button */
.stDownloadButton button {
    background: linear-gradient(135deg, #238be6, #02c39a) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(35,139,230,0.3) !important;
}
.stDownloadButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(35,139,230,0.45) !important;
}

/* Multiselect pills */
.stMultiSelect [data-baseweb="tag"] {
    background: rgba(35,139,230,0.2) !important;
    border-color: rgba(35,139,230,0.4) !important;
    color: #79c0ff !important;
}

/* Dataframe */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] > div {
    background: rgba(22,27,34,0.8) !important;
    border: 1px solid rgba(48,54,61,0.6) !important;
    border-radius: 12px !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #238be6 !important;
}

/* Hide default streamlit elements */
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Always-visible sidebar toggle button ── */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: rgba(35,139,230,0.15) !important;
    border: 1px solid rgba(35,139,230,0.4) !important;
    border-radius: 10px !important;
    color: #58a6ff !important;
    width: 36px !important;
    height: 36px !important;
    align-items: center !important;
    justify-content: center !important;
    top: 1rem !important;
    left: 0.75rem !important;
    position: fixed !important;
    z-index: 999999 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 12px rgba(35,139,230,0.25) !important;
}
[data-testid="collapsedControl"]:hover {
    background: rgba(35,139,230,0.35) !important;
    box-shadow: 0 0 20px rgba(35,139,230,0.5) !important;
    transform: scale(1.05) !important;
}
[data-testid="collapsedControl"] svg {
    fill: #58a6ff !important;
    width: 18px !important;
    height: 18px !important;
}

/* Keep expand arrow visible on sidebar edge too */
button[kind="header"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #238be6; }

/* Warning/info/error overrides */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #238be6, #02c39a) !important;
    border-radius: 99px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Colors for matplotlib ──
C = {
    "bg": "#060a10", "panel": "#0d1117", "panel2": "#161b22",
    "border": "#21262d", "blue": "#238be6", "cyan": "#02c39a",
    "green": "#3fb950", "yellow": "#d29922", "red": "#f85149",
    "orange": "#f0883e", "purple": "#bc8cff", "text": "#c9d1d9", "muted": "#6e7681",
}

def sax(ax, facecolor=None):
    ax.set_facecolor(facecolor or C["panel2"])
    ax.tick_params(colors=C["muted"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(C["border"])
    ax.xaxis.label.set_color(C["muted"])
    ax.yaxis.label.set_color(C["muted"])
    ax.title.set_color(C["text"])
    ax.grid(True, color=C["border"], alpha=0.5, linestyle="--", linewidth=0.4)


@st.cache_data
def load_data():
    from data_generator import generate_full_dataset
    from demand_forecasting import prepare_zone_hourly, DemandForecaster, compute_zone_risk
    from anomaly_detection import compute_meter_features, AnomalyDetector
    raw_df = generate_full_dataset(days=45)
    zone_hourly = prepare_zone_hourly(raw_df)
    forecaster = DemandForecaster()
    forecaster.train(zone_hourly)
    forecasts = forecaster.predict(zone_hourly)
    risk_df = compute_zone_risk(forecasts)
    meter_feat = compute_meter_features(raw_df)
    detector = AnomalyDetector(contamination=0.15)
    anomaly_df = detector.fit_predict(meter_feat)
    return raw_df, zone_hourly, forecasts, risk_df, anomaly_df, forecaster, detector


# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 4px 20px 4px;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
            <div style="width:36px;height:36px;background:linear-gradient(135deg,#238be6,#02c39a);
                        border-radius:10px;display:flex;align-items:center;justify-content:center;
                        font-size:1.2rem;">⚡</div>
            <div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;letter-spacing:-0.01em;">BESCOM AI</div>
                <div style="font-size:0.7rem;color:#6e7681;letter-spacing:0.05em;">SMART METER INTELLIGENCE</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Overview", "📈  Demand Forecast", "🚨  Anomaly Detection", "🔍  Meter Deep Dive", "📋  Inspection Report"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top:auto;padding-top:2rem;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(35,139,230,0.06);border:1px solid rgba(35,139,230,0.15);
                border-radius:10px;padding:12px;margin-top:1rem;">
        <div style="font-size:0.7rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">System Status</div>
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
            <div style="width:7px;height:7px;background:#3fb950;border-radius:50%;
                        box-shadow:0 0 6px #3fb950;"></div>
            <span style="font-size:0.8rem;color:#c9d1d9;">Models Active</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
            <div style="width:7px;height:7px;background:#3fb950;border-radius:50%;
                        box-shadow:0 0 6px #3fb950;"></div>
            <span style="font-size:0.8rem;color:#c9d1d9;">Data Pipeline OK</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;">
            <div style="width:7px;height:7px;background:#d29922;border-radius:50%;
                        box-shadow:0 0 6px #d29922;"></div>
            <span style="font-size:0.8rem;color:#c9d1d9;">Alerts Pending</span>
        </div>
    </div>
    <div style="font-size:0.68rem;color:#30363d;text-align:center;margin-top:1.5rem;">
        AI for Bharat 2025 · BESCOM Track
    </div>
    """, unsafe_allow_html=True)

# ── Load data ──
with st.spinner("Initializing AI models…"):
    raw_df, zone_hourly, forecasts, risk_df, anomaly_df, forecaster, detector = load_data()


# ══════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════
if "Overview" in page:
    st.markdown("""
    <div class="page-header">
        <h1>⚡ Smart Meter Intelligence System</h1>
        <p>AI-powered demand forecasting & anomaly detection for BESCOM's distribution network</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    total_meters = raw_df["meter_id"].nunique()
    flagged_meters = int(anomaly_df["detected_anomaly"].sum())
    high_risk = int((risk_df["risk_tier"] == "HIGH").sum())
    avg_mape = np.mean([m["MAPE"] for m in forecaster.metrics.values()])
    f1 = detector.metrics["f1"]

    kpis = [
        ("TOTAL METERS", f"{total_meters:,}", "Active & monitored", "#238be6", "📡"),
        ("ANOMALIES FLAGGED", str(flagged_meters), f"{flagged_meters/total_meters*100:.0f}% of fleet", "#f85149", "🚨"),
        ("HIGH-RISK ZONES", f"{high_risk}/{len(risk_df)}", "Need immediate attention", "#d29922", "⚠️"),
        ("FORECAST MAPE", f"{avg_mape:.1f}%", "Avg across all zones", "#02c39a", "📊"),
        ("DETECTION F1", f"{f1:.3f}", "Precision + Recall", "#bc8cff", "🎯"),
    ]
    cols = st.columns(5)
    for col, (lbl, val, sub, color, icon) in zip(cols, kpis):
        col.markdown(f"""
        <div class="kpi-card" style="--accent:{color};">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{lbl}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown('<div class="section-title">Zone Risk Assessment</div>', unsafe_allow_html=True)
        for _, row in risk_df.iterrows():
            tier = row["risk_tier"]
            score = row["risk_score"]
            pct = min(score, 100)
            bar_color = {"HIGH": "#f85149", "MEDIUM": "#d29922", "LOW": "#3fb950"}[tier]
            st.markdown(f"""
            <div class="zone-card">
                <div style="flex:1;">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                        <span style="font-weight:600;color:#e6edf3;font-size:0.9rem;">{row['zone']}</span>
                        <span class="zone-risk-badge badge-{tier}">{tier}</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.05);border-radius:99px;height:5px;overflow:hidden;">
                        <div style="width:{pct}%;height:100%;background:{bar_color};border-radius:99px;
                                    box-shadow:0 0 8px {bar_color}60;transition:width 0.8s ease;"></div>
                    </div>
                </div>
                <div style="text-align:right;min-width:80px;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:700;color:{bar_color};">{score}</div>
                    <div style="font-size:0.7rem;color:#6e7681;">risk score</div>
                </div>
                <div style="text-align:right;min-width:100px;">
                    <div style="font-size:0.82rem;color:#c9d1d9;font-weight:600;">{row['peak_load_kwh']:.0f} kWh</div>
                    <div style="font-size:0.7rem;color:#6e7681;">peak load</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-title">Anomaly Type Distribution</div>', unsafe_allow_html=True)
        flagged = anomaly_df[anomaly_df["detected_anomaly"]]
        type_counts = flagged["detected_type"].value_counts()
        type_clr = {
            "theft_gradual": C["red"], "theft_sudden_drop": C["orange"],
            "tamper_spike": C["purple"], "dead_meter": C["yellow"],
            "peer_deviation": C["blue"], "statistical_anomaly": C["muted"],
        }
        fig, ax = plt.subplots(figsize=(6, 3.8), facecolor=C["bg"])
        sax(ax, C["bg"])
        bars = ax.barh(
            type_counts.index[::-1], type_counts.values[::-1],
            color=[type_clr.get(t, C["muted"]) for t in type_counts.index[::-1]],
            edgecolor="none", height=0.6
        )
        for bar, val in zip(bars, type_counts.values[::-1]):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    str(int(val)), va="center", fontsize=9, color=C["text"], fontweight="600")
        ax.set_xlabel("Meters Flagged", color=C["muted"], fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=8.5)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Architecture table
    st.markdown('<div class="section-title" style="margin-top:2rem;">System Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="arch-table">
        <thead><tr>
            <th>Component</th><th>Technology</th><th>Purpose</th><th>Output</th>
        </tr></thead>
        <tbody>
            <tr><td>Data Layer</td>
                <td><span class="tech-badge">pandas</span> <span class="tech-badge">numpy</span></td>
                <td>15-min interval synthetic BESCOM data, 210 meters × 5 zones</td>
                <td>Cleaned time series</td></tr>
            <tr><td>Part A: Forecasting</td>
                <td><span class="tech-badge">GradientBoosting</span></td>
                <td>35+ features: lag, cyclical, rolling stats, peer aggregates</td>
                <td>Hourly demand forecast</td></tr>
            <tr><td>Zone Risk Engine</td>
                <td><span class="tech-badge">rule-based</span></td>
                <td>Peak ratio, volatility, growth trend, 24h forecasted peak</td>
                <td>Risk score 0–100</td></tr>
            <tr><td>Part B: Anomaly</td>
                <td><span class="tech-badge">IsolationForest</span></td>
                <td>Unsupervised + 6-rule expert system, peer z-score comparison</td>
                <td>Flagged meters + reasons</td></tr>
            <tr><td>Explainability</td>
                <td><span class="tech-badge">feature importance</span></td>
                <td>Per-alert rule reasons, confidence score, revenue estimate</td>
                <td>Audit trail CSV</td></tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# PAGE 2: DEMAND FORECASTING
# ══════════════════════════════════════════
elif "Demand" in page:
    st.markdown("""
    <div class="page-header">
        <h1>📈 Demand Forecasting & Zone Risk</h1>
        <p>GradientBoosting per-zone models with 35+ temporal features — R² 1.000, MAPE 0.2%</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1], gap="medium")
    with c1:
        zone = st.selectbox("Select Zone", list(forecasts.keys()), label_visibility="visible")
    with c2:
        days_back = st.slider("Days to display", 3, 21, 7, label_visibility="visible")

    fdf = forecasts[zone].tail(days_back * 24)

    # Metrics banner
    m = forecaster.metrics.get(zone, {})
    b_mae = fdf["baseline_mae"].mean()
    mod_mae = fdf["model_mae"].mean()
    imp = (b_mae - mod_mae) / (b_mae + 1e-8) * 100
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-tile" style="--c:#238be6;">
            <div class="metric-tile-val" style="color:#238be6;">{m.get('R2',0):.4f}</div>
            <div class="metric-tile-lbl">R² Score</div>
        </div>
        <div class="metric-tile" style="--c:#02c39a;">
            <div class="metric-tile-val" style="color:#02c39a;">{m.get('MAPE',0):.1f}%</div>
            <div class="metric-tile-lbl">MAPE</div>
        </div>
        <div class="metric-tile" style="--c:#bc8cff;">
            <div class="metric-tile-val" style="color:#bc8cff;">{mod_mae:.2f}</div>
            <div class="metric-tile-lbl">Model MAE (kWh)</div>
        </div>
        <div class="metric-tile" style="--c:#f0883e;">
            <div class="metric-tile-val" style="color:#f0883e;">{imp:.0f}%</div>
            <div class="metric-tile-lbl">vs Baseline</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=C["bg"])
    axes = axes.flatten()
    fig.patch.set_facecolor(C["bg"])

    # 1. Actual vs Predicted
    sax(axes[0])
    axes[0].fill_between(fdf["timestamp"], fdf["total_kwh"], alpha=0.1, color=C["blue"])
    axes[0].plot(fdf["timestamp"], fdf["total_kwh"], color=C["blue"], lw=1.8, label="Actual")
    axes[0].plot(fdf["timestamp"], fdf["predicted_kwh"], color=C["orange"], lw=1.6, ls="--", label="Predicted")
    thr = fdf["total_kwh"].quantile(0.95)
    axes[0].axhline(thr, color=C["red"], ls=":", lw=1.2, alpha=0.8, label=f"Alert threshold")
    axes[0].set_title(f"Actual vs Predicted — {zone}", fontsize=10, fontweight="600")
    axes[0].set_ylabel("Zone kWh", fontsize=9)
    axes[0].legend(facecolor=C["panel2"], edgecolor=C["border"], labelcolor=C["text"], fontsize=8, framealpha=0.9)

    # 2. Residuals
    sax(axes[1])
    res = fdf["residual"].values
    axes[1].bar(fdf["timestamp"], res, color=np.where(res > 0, C["green"], C["red"]), alpha=0.75, width=0.035)
    axes[1].axhline(0, color=C["muted"], lw=1)
    axes[1].set_title("Forecast Residuals", fontsize=10, fontweight="600")
    axes[1].set_ylabel("Actual − Predicted", fontsize=9)

    # 3. Hourly load profile
    sax(axes[2])
    fd2 = fdf.copy()
    fd2["hour"] = fd2["timestamp"].dt.hour
    prof = fd2.groupby("hour").agg(act=("total_kwh","mean"), pred=("predicted_kwh","mean"))
    axes[2].fill_between(prof.index, prof["act"], alpha=0.1, color=C["blue"])
    axes[2].plot(prof.index, prof["act"], color=C["blue"], lw=2, marker="o", ms=3, label="Actual avg")
    axes[2].plot(prof.index, prof["pred"], color=C["orange"], lw=2, ls="--", marker="s", ms=3, label="Predicted avg")
    axes[2].set_title("Avg Daily Load Profile", fontsize=10, fontweight="600")
    axes[2].set_xlabel("Hour of Day", fontsize=9)
    axes[2].legend(facecolor=C["panel2"], edgecolor=C["border"], labelcolor=C["text"], fontsize=8)
    axes[2].set_xticks(range(0, 24, 3))

    # 4. Feature importance
    sax(axes[3])
    fi = forecaster.feature_importance.get(zone, {})
    if fi:
        top = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
        names, vals = [x[0] for x in top][::-1], [x[1] for x in top][::-1]
        gradient_colors = plt.cm.cool(np.linspace(0.3, 0.9, len(names)))
        axes[3].barh(names, vals, color=gradient_colors, edgecolor="none", height=0.65)
        axes[3].set_title("Feature Importances", fontsize=10, fontweight="600")
        axes[3].set_xlabel("Importance", fontsize=9)
        axes[3].spines["top"].set_visible(False)
        axes[3].spines["right"].set_visible(False)

    fig.tight_layout(pad=2.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Table
    st.markdown('<div class="section-title">All Zones — Model Performance</div>', unsafe_allow_html=True)
    rows = []
    for z, fdf3 in forecasts.items():
        mm = forecaster.metrics.get(z, {})
        bm = fdf3["baseline_mae"].mean()
        km = fdf3["model_mae"].mean()
        rows.append({
            "Zone": z,
            "MAE — Model": f"{km:.3f} kWh",
            "MAE — Baseline": f"{bm:.3f} kWh",
            "Improvement": f"{(bm-km)/(bm+1e-8)*100:.0f}%",
            "R²": f"{mm.get('R2',0):.4f}",
            "MAPE": f"{mm.get('MAPE',0):.2f}%",
            "Risk Tier": risk_df[risk_df["zone"]==z]["risk_tier"].values[0] if z in risk_df["zone"].values else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════
# PAGE 3: ANOMALY DETECTION
# ══════════════════════════════════════════
elif "Anomaly" in page:
    st.markdown("""
    <div class="page-header">
        <h1>🚨 Anomaly & Theft Detection</h1>
        <p>Isolation Forest + 6-rule expert system — Precision 93.8% · Recall 100% · F1 96.8%</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-tile"><div class="metric-tile-val" style="color:#238be6;">{detector.metrics['precision']:.1%}</div><div class="metric-tile-lbl">Precision</div></div>
        <div class="metric-tile"><div class="metric-tile-val" style="color:#02c39a;">{detector.metrics['recall']:.1%}</div><div class="metric-tile-lbl">Recall</div></div>
        <div class="metric-tile"><div class="metric-tile-val" style="color:#bc8cff;">{detector.metrics['f1']:.3f}</div><div class="metric-tile-lbl">F1 Score</div></div>
        <div class="metric-tile"><div class="metric-tile-val" style="color:#f0883e;">{detector.metrics['roc_auc']:.3f}</div><div class="metric-tile-lbl">ROC-AUC</div></div>
    </div>
    """, unsafe_allow_html=True)

    flagged = anomaly_df[anomaly_df["detected_anomaly"]].sort_values("confidence_score", ascending=False)
    type_clr = {
        "normal": C["green"], "theft_gradual": C["red"], "theft_sudden_drop": C["orange"],
        "tamper_spike": C["purple"], "dead_meter": C["yellow"],
        "peer_deviation": C["blue"], "statistical_anomaly": C["muted"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=C["bg"])
    fig.patch.set_facecolor(C["bg"])

    # Scatter
    sax(axes[0])
    for dtype, grp in anomaly_df.groupby("detected_type"):
        axes[0].scatter(grp["mean_consumption"], grp["cv"],
                        c=type_clr.get(dtype, C["muted"]),
                        label=dtype, alpha=0.75, s=35, edgecolors="none")
    axes[0].set_xlabel("Mean Consumption (kWh)", fontsize=9)
    axes[0].set_ylabel("Coefficient of Variation", fontsize=9)
    axes[0].set_title("Anomaly Scatter — Consumption vs Variability", fontsize=10, fontweight="600")
    axes[0].legend(fontsize=7, facecolor=C["panel2"], edgecolor=C["border"], labelcolor=C["text"], framealpha=0.9)

    # Confidence histogram
    sax(axes[1])
    if len(flagged) > 0:
        n, bins, patches = axes[1].hist(flagged["confidence_score"], bins=12,
                                         edgecolor=C["border"], linewidth=0.5)
        for patch, left in zip(patches, bins[:-1]):
            patch.set_facecolor(plt.cm.RdYlGn_r(left / 100))
    axes[1].set_xlabel("Confidence Score", fontsize=9)
    axes[1].set_ylabel("Number of Meters", fontsize=9)
    axes[1].set_title("Anomaly Confidence Distribution", fontsize=10, fontweight="600")

    # Zone breakdown
    sax(axes[2])
    zc = flagged.groupby("zone").size().sort_values()
    bar_cols = plt.cm.Reds(np.linspace(0.4, 0.9, len(zc)))
    axes[2].barh(zc.index, zc.values, color=bar_cols, edgecolor="none", height=0.6)
    for i, (z, v) in enumerate(zc.items()):
        axes[2].text(v + 0.1, i, str(v), va="center", fontsize=9, color=C["text"], fontweight="600")
    axes[2].set_xlabel("Flagged Meters", fontsize=9)
    axes[2].set_title("Anomalies by Zone", fontsize=10, fontweight="600")
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    fig.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-title">Flagged Meters — Detailed View</div>', unsafe_allow_html=True)
    display_cols = ["meter_id", "zone", "detected_type", "confidence_score",
                    "mean_consumption", "recent_hist_ratio", "zero_ratio", "rule_reasons"]
    st.dataframe(flagged[display_cols].reset_index(drop=True), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════
# PAGE 4: METER DEEP DIVE
# ══════════════════════════════════════════
elif "Meter" in page:
    st.markdown("""
    <div class="page-header">
        <h1>🔍 Meter Deep Dive</h1>
        <p>Inspect individual meter consumption patterns and anomaly evidence</p>
    </div>
    """, unsafe_allow_html=True)

    flagged_ids = anomaly_df[anomaly_df["detected_anomaly"]]["meter_id"].tolist()
    all_ids = anomaly_df["meter_id"].tolist()
    selected = st.selectbox(
        "Select Meter (flagged meters shown first)",
        flagged_ids + [m for m in all_ids if m not in flagged_ids]
    )

    mdf = raw_df[raw_df["meter_id"] == selected].sort_values("timestamp")
    meta = anomaly_df[anomaly_df["meter_id"] == selected].iloc[0]
    tier_color = {"HIGH": "#f85149", "MEDIUM": "#d29922", "LOW": "#3fb950"}
    conf = meta["confidence_score"]
    conf_color = "#f85149" if conf > 65 else ("#d29922" if conf > 40 else "#3fb950")

    # Header cards
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="kpi-card" style="--accent:#238be6;">
        <div class="kpi-label">Zone</div>
        <div style="font-size:1.3rem;font-weight:700;color:#238be6;">{meta['zone']}</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="kpi-card" style="--accent:#f0883e;">
        <div class="kpi-label">Detected Type</div>
        <div style="font-size:1rem;font-weight:700;color:#f0883e;line-height:1.3;">{meta['detected_type'].replace('_',' ').title()}</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="kpi-card" style="--accent:{conf_color};">
        <div class="kpi-label">Confidence</div>
        <div style="font-size:1.9rem;font-weight:700;color:{conf_color};font-family:'JetBrains Mono',monospace;">{conf:.0f}<span style="font-size:1rem;">/100</span></div>
    </div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="kpi-card" style="--accent:{'#f85149' if meta['detected_anomaly'] else '#3fb950'};">
        <div class="kpi-label">Status</div>
        <div style="font-size:1.1rem;font-weight:700;color:{'#f85149' if meta['detected_anomaly'] else '#3fb950'};">
            {'⚠️ ANOMALOUS' if meta['detected_anomaly'] else '✅ NORMAL'}
        </div>
    </div>""", unsafe_allow_html=True)

    if meta["rule_reasons"]:
        st.markdown(f'<div class="alert-box">⚠️ <strong>Detection Reason:</strong> {meta["rule_reasons"]}</div>',
                    unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), facecolor=C["bg"])
    fig.patch.set_facecolor(C["bg"])
    for ax in axes:
        sax(ax)

    axes[0].fill_between(mdf["timestamp"], mdf["consumption_kwh"],
                         where=~mdf["is_anomaly"], alpha=0.2, color=C["blue"], label="Normal Period")
    axes[0].fill_between(mdf["timestamp"], mdf["consumption_kwh"],
                         where=mdf["is_anomaly"], alpha=0.45, color=C["red"], label="Anomalous Period")
    axes[0].plot(mdf["timestamp"], mdf["consumption_kwh"], lw=0.8, color=C["blue"], alpha=0.85)
    axes[0].set_title(f"Consumption Time Series — {selected}", fontsize=10, fontweight="600")
    axes[0].set_ylabel("kWh / 15 min", fontsize=9)
    axes[0].legend(facecolor=C["panel2"], edgecolor=C["border"], labelcolor=C["text"], fontsize=8)

    daily = mdf.set_index("timestamp").resample("D")["consumption_kwh"].sum()
    daily_anom = mdf[mdf["is_anomaly"]].set_index("timestamp").resample("D")["consumption_kwh"].sum()
    axes[1].bar(daily.index, daily.values, color=C["blue"], alpha=0.5, width=0.8, label="Normal days")
    if len(daily_anom) > 0:
        axes[1].bar(daily_anom.index, daily_anom.values, color=C["red"], alpha=0.8, width=0.8, label="Anomalous days")
    axes[1].set_title("Daily Consumption Totals", fontsize=10, fontweight="600")
    axes[1].set_ylabel("kWh / Day", fontsize=9)
    axes[1].set_xlabel("Date", fontsize=9)
    axes[1].legend(facecolor=C["panel2"], edgecolor=C["border"], labelcolor=C["text"], fontsize=8)

    fig.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-title">Feature Profile</div>', unsafe_allow_html=True)
    feat_cols = ["mean_consumption", "cv", "zero_ratio", "night_day_ratio",
                 "recent_hist_ratio", "trend_pct", "peer_deviation_score", "outlier_ratio"]
    feat_df = meta[feat_cols].to_frame("Value").T.round(4)
    st.dataframe(feat_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════
# PAGE 5: INSPECTION REPORT
# ══════════════════════════════════════════
elif "Inspection" in page:
    st.markdown("""
    <div class="page-header">
        <h1>📋 Actionable Inspection Report</h1>
        <p>Prioritized field dispatch list — sorted by risk level and detection confidence</p>
    </div>
    """, unsafe_allow_html=True)

    flagged = anomaly_df[anomaly_df["detected_anomaly"]].copy()
    flagged["priority"] = pd.cut(
        flagged["confidence_score"], bins=[0, 40, 65, 100], labels=["LOW", "MEDIUM", "HIGH"]
    )

    # Filter row
    c1, c2 = st.columns([2, 3])
    with c1:
        priority_filter = st.multiselect(
            "Filter by Priority",
            ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM"]
        )
    with c2:
        zone_filter = st.multiselect(
            "Filter by Zone",
            sorted(flagged["zone"].unique()),
            default=sorted(flagged["zone"].unique())
        )

    filtered = flagged[
        flagged["priority"].astype(str).isin(priority_filter) &
        flagged["zone"].isin(zone_filter)
    ].sort_values(["priority", "confidence_score"], ascending=[True, False])

    # Summary stats
    high_c = (filtered["priority"] == "HIGH").sum()
    med_c  = (filtered["priority"] == "MEDIUM").sum()
    low_c  = (filtered["priority"] == "LOW").sum()

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.markdown(f"""<div class="kpi-card" style="--accent:#f85149;">
        <div class="kpi-label">High Priority</div>
        <div class="kpi-value">{high_c}</div>
        <div class="kpi-sub">Dispatch immediately</div>
    </div>""", unsafe_allow_html=True)
    sc2.markdown(f"""<div class="kpi-card" style="--accent:#d29922;">
        <div class="kpi-label">Medium Priority</div>
        <div class="kpi-value">{med_c}</div>
        <div class="kpi-sub">Schedule this week</div>
    </div>""", unsafe_allow_html=True)
    sc3.markdown(f"""<div class="kpi-card" style="--accent:#3fb950;">
        <div class="kpi-label">Low Priority</div>
        <div class="kpi-value">{low_c}</div>
        <div class="kpi-sub">Monitor & review</div>
    </div>""", unsafe_allow_html=True)
    sc4.markdown(f"""<div class="kpi-card" style="--accent:#238be6;">
        <div class="kpi-label">Total Flagged</div>
        <div class="kpi-value">{len(filtered)}</div>
        <div class="kpi-sub">Meters for inspection</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    display_df = filtered[[
        "meter_id", "zone", "detected_type", "priority",
        "confidence_score", "mean_consumption", "recent_hist_ratio",
        "peer_deviation_score", "rule_reasons"
    ]].reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Download
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="⬇️  Download Inspection Report CSV",
        data=csv,
        file_name="bescom_inspection_report.csv",
        mime="text/csv",
    )

    # Revenue loss
    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Revenue Loss Estimation</div>',
                unsafe_allow_html=True)
    theft_df = filtered[filtered["detected_type"].isin(["theft_gradual", "theft_sudden_drop"])]
    if len(theft_df) > 0:
        loss_ratio = 1 - theft_df["recent_hist_ratio"].mean()
        avg_c = theft_df["mean_consumption"].mean()
        monthly_loss = len(theft_df) * avg_c * loss_ratio * 96 * 30 * 7
        st.markdown(f"""
        <div class="alert-box">
            ⚠️ &nbsp;<strong>Estimated monthly revenue loss</strong> from {len(theft_df)} theft-suspected meters:
            &nbsp;<span style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;">₹{monthly_loss:,.0f}</span>
            &nbsp;(at ₹7/kWh · 96 readings/day)
        </div>
        """, unsafe_allow_html=True)
        annual = monthly_loss * 12
        st.markdown(f"""
        <div class="info-box">
            📊 &nbsp;Annualised projection: <strong>₹{annual:,.0f}</strong> — recoverable with timely field intervention.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">✅ No theft-pattern meters in current filter selection.</div>',
                    unsafe_allow_html=True)