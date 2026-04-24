"""
BESCOM Smart Meter Intelligence — Streamlit Dashboard
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="BESCOM Smart Meter AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark theme CSS ───
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stMetric { background-color: #161b22; border-radius: 8px; padding: 12px; }
    .stMetric label { color: #8b949e !important; }
    .stMetric div { color: #c9d1d9 !important; }
    h1, h2, h3 { color: #58a6ff !important; }
    .risk-high { color: #f85149; font-weight: bold; }
    .risk-medium { color: #d29922; font-weight: bold; }
    .risk-low { color: #3fb950; font-weight: bold; }
    div[data-testid="stDataFrame"] { background: #161b22; }
    .sidebar .sidebar-content { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "bg": "#0d1117", "panel": "#161b22", "border": "#30363d",
    "blue": "#58a6ff", "green": "#3fb950", "yellow": "#d29922",
    "red": "#f85149", "orange": "#f0883e", "purple": "#bc8cff",
    "text": "#c9d1d9", "muted": "#8b949e",
}


def style_ax(ax):
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["border"])
    ax.xaxis.label.set_color(COLORS["muted"])
    ax.yaxis.label.set_color(COLORS["muted"])
    ax.title.set_color(COLORS["text"])
    ax.grid(True, color=COLORS["border"], alpha=0.4, linestyle="--", linewidth=0.5)


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


# ─── Sidebar ───
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/2/2a/BESCOM_Logo.png/200px-BESCOM_Logo.png",
                 width=100) if False else None
st.sidebar.title("⚡ BESCOM AI System")
st.sidebar.markdown("**Smart Meter Intelligence**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📈 Demand Forecasting", "🚨 Anomaly Detection", "🔍 Meter Deep Dive", "📋 Inspection Report"]
)

# ─── Load Data ───
with st.spinner("Initializing AI models... (first run takes ~30s)"):
    raw_df, zone_hourly, forecasts, risk_df, anomaly_df, forecaster, detector = load_data()

# ══════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════
if page == "🏠 Overview":
    st.title("⚡ BESCOM Smart Meter Intelligence System")
    st.markdown("**AI-powered demand forecasting and anomaly detection for BESCOM's distribution network**")
    st.markdown("---")

    # KPI cards
    col1, col2, col3, col4, col5 = st.columns(5)
    total_meters = raw_df["meter_id"].nunique()
    flagged_meters = anomaly_df["detected_anomaly"].sum()
    high_risk_zones = (risk_df["risk_tier"] == "HIGH").sum()
    avg_mape = np.mean([m["MAPE"] for m in forecaster.metrics.values()])
    f1 = detector.metrics["f1"]

    col1.metric("Total Meters", f"{total_meters:,}", "Active")
    col2.metric("Flagged Anomalies", f"{flagged_meters}", f"~{flagged_meters/total_meters*100:.0f}% of meters")
    col3.metric("High-Risk Zones", f"{high_risk_zones}/{len(risk_df)}", "Need attention")
    col4.metric("Forecast MAPE", f"{avg_mape:.1f}%", "Across all zones")
    col5.metric("Detection F1", f"{f1:.3f}", "Precision+Recall")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🗺️ Zone Risk Assessment")
        risk_colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        for _, row in risk_df.iterrows():
            icon = risk_colors[row["risk_tier"]]
            st.markdown(
                f"{icon} **{row['zone']}** — Risk Score: `{row['risk_score']}` | "
                f"Peak: `{row['peak_load_kwh']:.0f} kWh` | "
                f"Trend: `{row['load_trend_pct']:+.1f}%`"
            )

    with col_b:
        st.subheader("📊 Anomaly Type Breakdown")
        flagged = anomaly_df[anomaly_df["detected_anomaly"]]
        type_counts = flagged["detected_type"].value_counts()

        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor=COLORS["bg"])
        style_ax(ax)
        type_colors_map = {
            "theft_gradual": COLORS["red"], "theft_sudden_drop": COLORS["orange"],
            "tamper_spike": COLORS["purple"], "dead_meter": COLORS["yellow"],
            "peer_deviation": COLORS["blue"], "statistical_anomaly": COLORS["muted"],
        }
        bars = ax.bar(type_counts.index, type_counts.values,
                      color=[type_colors_map.get(t, COLORS["muted"]) for t in type_counts.index],
                      edgecolor=COLORS["border"])
        ax.set_xticklabels(type_counts.index, rotation=25, ha="right", fontsize=7)
        ax.set_title("Detected Anomaly Types", color=COLORS["text"])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    str(int(bar.get_height())), ha="center", fontsize=8, color=COLORS["text"])
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("ℹ️ System Architecture")
    st.markdown("""
    | Component | Technology | Purpose |
    |-----------|-----------|---------|
    | **Data Layer** | Synthetic BESCOM-like data (15-min intervals) | 210 meters × 5 zones × 90 days |
    | **Part A: Forecasting** | Gradient Boosting + Lag/Cyclical features | Hourly/day-ahead demand prediction |
    | **Zone Risk Scoring** | Rule-based composite scoring | Peak ratio, volatility, trend, forecast |
    | **Part B: Anomaly** | Isolation Forest + Expert Rules | Statistical + pattern-based detection |
    | **Explainability** | Feature importance + rule reasons | Actionable audit trail per meter |
    | **Dashboard** | Streamlit | Real-time decision support interface |
    """)


# ══════════════════════════════════════════
# PAGE 2: DEMAND FORECASTING
# ══════════════════════════════════════════
elif page == "📈 Demand Forecasting":
    st.title("📈 Demand Forecasting & Zone Risk")

    zone = st.selectbox("Select Zone", list(forecasts.keys()))
    days_back = st.slider("Days to display", 3, 21, 7)

    fdf = forecasts[zone].tail(days_back * 24)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=COLORS["bg"])
    axes = axes.flatten()

    # Actual vs Predicted
    style_ax(axes[0])
    axes[0].fill_between(fdf["timestamp"], fdf["total_kwh"], alpha=0.15, color=COLORS["blue"])
    axes[0].plot(fdf["timestamp"], fdf["total_kwh"], color=COLORS["blue"], lw=1.5, label="Actual")
    axes[0].plot(fdf["timestamp"], fdf["predicted_kwh"], color=COLORS["orange"],
                 lw=1.5, ls="--", label="Predicted")
    threshold = fdf["total_kwh"].quantile(0.95)
    axes[0].axhline(threshold, color=COLORS["red"], ls=":", lw=1.2, label=f"Alert threshold ({threshold:.0f} kWh)")
    axes[0].set_title(f"Actual vs Predicted — {zone}", fontsize=10)
    axes[0].set_ylabel("Zone kWh")
    axes[0].legend(facecolor=COLORS["panel"], edgecolor=COLORS["border"],
                   labelcolor=COLORS["text"], fontsize=7)

    # Residuals
    style_ax(axes[1])
    axes[1].bar(fdf["timestamp"], fdf["residual"],
                color=np.where(fdf["residual"] > 0, COLORS["green"], COLORS["red"]), alpha=0.7, width=0.03)
    axes[1].axhline(0, color=COLORS["muted"], lw=0.8)
    axes[1].set_title("Forecast Residuals", fontsize=10)
    axes[1].set_ylabel("Actual - Predicted (kWh)")

    # Hourly avg profile
    style_ax(axes[2])
    fdf_copy = fdf.copy()
    fdf_copy["hour"] = fdf_copy["timestamp"].dt.hour
    profile = fdf_copy.groupby("hour").agg(
        actual_mean=("total_kwh", "mean"),
        pred_mean=("predicted_kwh", "mean"),
    )
    axes[2].plot(profile.index, profile["actual_mean"], color=COLORS["blue"], lw=2, label="Actual avg")
    axes[2].plot(profile.index, profile["pred_mean"], color=COLORS["orange"], lw=2, ls="--", label="Predicted avg")
    axes[2].set_title("Avg Hourly Load Profile", fontsize=10)
    axes[2].set_xlabel("Hour of Day")
    axes[2].legend(facecolor=COLORS["panel"], edgecolor=COLORS["border"],
                   labelcolor=COLORS["text"], fontsize=7)

    # Feature importance
    style_ax(axes[3])
    fi = forecaster.feature_importance.get(zone, {})
    if fi:
        top_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
        names = [x[0] for x in top_fi]
        vals = [x[1] for x in top_fi]
        axes[3].barh(names[::-1], vals[::-1], color=COLORS["purple"], edgecolor=COLORS["border"])
        axes[3].set_title("Top Feature Importances", fontsize=10)

    fig.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=True)

    # Metrics table
    st.subheader("Model Performance vs Baseline")
    rows = []
    for z, fdf3 in forecasts.items():
        m = forecaster.metrics.get(z, {})
        baseline_mae = fdf3["baseline_mae"].mean()
        model_mae = fdf3["model_mae"].mean()
        improvement = (baseline_mae - model_mae) / (baseline_mae + 1e-8) * 100
        rows.append({
            "Zone": z, "MAE (Model)": f"{model_mae:.2f}",
            "MAE (Baseline)": f"{baseline_mae:.2f}",
            "Improvement": f"{improvement:.0f}%",
            "R²": f"{m.get('R2', 0):.4f}",
            "MAPE": f"{m.get('MAPE', 0):.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Zone Risk Details")
    st.dataframe(risk_df, use_container_width=True)


# ══════════════════════════════════════════
# PAGE 3: ANOMALY DETECTION
# ══════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Anomaly & Theft Detection")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{detector.metrics['precision']:.1%}")
    col2.metric("Recall", f"{detector.metrics['recall']:.1%}")
    col3.metric("F1 Score", f"{detector.metrics['f1']:.3f}")
    col4.metric("ROC-AUC", f"{detector.metrics['roc_auc']:.3f}")

    flagged = anomaly_df[anomaly_df["detected_anomaly"]].sort_values("confidence_score", ascending=False)
    normal = anomaly_df[~anomaly_df["detected_anomaly"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=COLORS["bg"])

    # Scatter
    style_ax(axes[0])
    type_colors_map = {
        "normal": COLORS["green"], "theft_gradual": COLORS["red"],
        "theft_sudden_drop": COLORS["orange"], "tamper_spike": COLORS["purple"],
        "dead_meter": COLORS["yellow"], "peer_deviation": COLORS["blue"],
        "statistical_anomaly": COLORS["muted"],
    }
    for dtype, grp in anomaly_df.groupby("detected_type"):
        axes[0].scatter(grp["mean_consumption"], grp["cv"],
                        c=type_colors_map.get(dtype, COLORS["muted"]),
                        label=dtype, alpha=0.7, s=30)
    axes[0].set_xlabel("Mean Consumption (kWh)")
    axes[0].set_ylabel("CV (Variability)")
    axes[0].set_title("Consumption vs Variability", fontsize=10)
    axes[0].legend(fontsize=6, facecolor=COLORS["panel"], edgecolor=COLORS["border"],
                   labelcolor=COLORS["text"])

    # Confidence hist
    style_ax(axes[1])
    if len(flagged) > 0:
        axes[1].hist(flagged["confidence_score"], bins=12, color=COLORS["red"],
                     edgecolor=COLORS["border"], alpha=0.8)
    axes[1].set_xlabel("Confidence Score")
    axes[1].set_title("Anomaly Confidence Distribution", fontsize=10)

    # By zone
    style_ax(axes[2])
    zone_counts = flagged.groupby("zone").size().sort_values()
    axes[2].barh(zone_counts.index, zone_counts.values,
                 color=COLORS["red"], edgecolor=COLORS["border"])
    axes[2].set_xlabel("Count")
    axes[2].set_title("Flagged Meters by Zone", fontsize=10)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("🚩 Flagged Meters")
    display_cols = ["meter_id", "zone", "detected_type", "confidence_score",
                    "mean_consumption", "recent_hist_ratio", "zero_ratio", "rule_reasons"]
    st.dataframe(flagged[display_cols].reset_index(drop=True), use_container_width=True)


# ══════════════════════════════════════════
# PAGE 4: METER DEEP DIVE
# ══════════════════════════════════════════
elif page == "🔍 Meter Deep Dive":
    st.title("🔍 Meter Deep Dive Analysis")

    flagged_ids = anomaly_df[anomaly_df["detected_anomaly"]]["meter_id"].tolist()
    all_ids = anomaly_df["meter_id"].tolist()
    selected = st.selectbox("Select Meter", flagged_ids + [m for m in all_ids if m not in flagged_ids])

    mdf = raw_df[raw_df["meter_id"] == selected].sort_values("timestamp")
    meta = anomaly_df[anomaly_df["meter_id"] == selected].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Zone", meta["zone"])
    col2.metric("Detected Type", meta["detected_type"])
    col3.metric("Confidence", f"{meta['confidence_score']:.0f}/100")
    col4.metric("Anomalous", "✅ YES" if meta["detected_anomaly"] else "✅ NO")

    if meta["rule_reasons"]:
        st.warning(f"**Rule-based alert:** {meta['rule_reasons']}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), facecolor=COLORS["bg"])
    for ax in axes:
        style_ax(ax)

    axes[0].fill_between(mdf["timestamp"], mdf["consumption_kwh"],
                         where=~mdf["is_anomaly"], alpha=0.25, color=COLORS["blue"], label="Normal")
    axes[0].fill_between(mdf["timestamp"], mdf["consumption_kwh"],
                         where=mdf["is_anomaly"], alpha=0.5, color=COLORS["red"], label="Anomalous")
    axes[0].plot(mdf["timestamp"], mdf["consumption_kwh"], lw=0.7, color=COLORS["blue"], alpha=0.8)
    axes[0].set_title(f"Consumption Time Series — {selected}", fontsize=10)
    axes[0].set_ylabel("kWh / 15min")
    axes[0].legend(facecolor=COLORS["panel"], edgecolor=COLORS["border"],
                   labelcolor=COLORS["text"], fontsize=8)

    daily = mdf.set_index("timestamp").resample("D")["consumption_kwh"].sum()
    axes[1].bar(daily.index, daily.values, color=COLORS["blue"], alpha=0.6, width=0.8)
    axes[1].set_title("Daily Totals", fontsize=10)
    axes[1].set_ylabel("kWh / Day")
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Feature Profile")
    feature_cols = ["mean_consumption", "cv", "zero_ratio", "night_day_ratio",
                    "recent_hist_ratio", "trend_pct", "peer_deviation_score",
                    "outlier_ratio", "max_drop_pct"]
    st.dataframe(meta[feature_cols].to_frame("value").T, use_container_width=True)


# ══════════════════════════════════════════
# PAGE 5: INSPECTION REPORT
# ══════════════════════════════════════════
elif page == "📋 Inspection Report":
    st.title("📋 Actionable Inspection Report")
    st.markdown("Prioritized list of meters for field inspection, sorted by risk level and confidence.")

    priority_filter = st.multiselect(
        "Filter by Priority",
        ["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM"]
    )

    flagged = anomaly_df[anomaly_df["detected_anomaly"]].copy()
    flagged["priority"] = pd.cut(
        flagged["confidence_score"], bins=[0, 40, 65, 100], labels=["LOW", "MEDIUM", "HIGH"]
    )

    filtered = flagged[flagged["priority"].astype(str).isin(priority_filter)]
    filtered = filtered.sort_values(["priority", "confidence_score"], ascending=[True, False])

    st.metric("Meters to inspect", len(filtered))

    display_df = filtered[[
        "meter_id", "zone", "detected_type", "priority",
        "confidence_score", "mean_consumption", "recent_hist_ratio",
        "peer_deviation_score", "rule_reasons"
    ]].reset_index(drop=True)

    st.dataframe(display_df, use_container_width=True)

    # Download CSV
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Inspection Report CSV",
        data=csv,
        file_name="bescom_inspection_report.csv",
        mime="text/csv"
    )

    st.subheader("Revenue Loss Estimation")
    flagged_theft = flagged[flagged["detected_type"].isin(["theft_gradual", "theft_sudden_drop"])]
    if len(flagged_theft) > 0:
        avg_loss_ratio = 1 - flagged_theft["recent_hist_ratio"].mean()
        avg_consumption = flagged_theft["mean_consumption"].mean()
        # 15-min intervals, 96/day, 30 days, ₹7/kWh
        est_monthly_loss = len(flagged_theft) * avg_consumption * avg_loss_ratio * 96 * 30 * 7
        st.error(f"⚠️ Estimated monthly revenue loss from {len(flagged_theft)} theft-suspected meters: "
                 f"**₹{est_monthly_loss:,.0f}** (at ₹7/kWh)")
    else:
        st.success("No theft-pattern meters detected.")
