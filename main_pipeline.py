"""
BESCOM Smart Meter Intelligence System — Main Pipeline
Runs: data generation → forecasting → anomaly detection → output reports
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# ── Imports ──
from data_generator import generate_full_dataset
from demand_forecasting import prepare_zone_hourly, DemandForecaster, compute_zone_risk
from anomaly_detection import compute_meter_features, AnomalyDetector


# ─────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────

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
    ax.grid(True, color=COLORS["border"], alpha=0.5, linestyle="--", linewidth=0.5)


def plot_demand_forecast(forecasts, risk_df, zone="Koramangala"):
    """Plot actual vs predicted demand + zone risk."""
    fig = plt.figure(figsize=(18, 12), facecolor=COLORS["bg"])
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Forecast for a zone (last 7 days)
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1)
    fdf = forecasts[zone].tail(7 * 24)
    ax1.fill_between(fdf["timestamp"], fdf["total_kwh"], alpha=0.2, color=COLORS["blue"])
    ax1.plot(fdf["timestamp"], fdf["total_kwh"], color=COLORS["blue"], lw=1.5, label="Actual")
    ax1.plot(fdf["timestamp"], fdf["predicted_kwh"], color=COLORS["orange"],
             lw=1.5, ls="--", label="Predicted")
    ax1.axhline(fdf["total_kwh"].quantile(0.95), color=COLORS["red"],
                ls=":", lw=1, label="95th pctile (alert threshold)")
    ax1.set_title(f"Demand Forecast — {zone} (Last 7 Days)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Total kWh (zone)")
    ax1.legend(facecolor=COLORS["panel"], edgecolor=COLORS["border"],
               labelcolor=COLORS["text"], fontsize=8)

    # 2. Risk scores bar chart
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2)
    risk_colors = {
        "HIGH": COLORS["red"], "MEDIUM": COLORS["yellow"], "LOW": COLORS["green"]
    }
    bars = ax2.barh(
        risk_df["zone"], risk_df["risk_score"],
        color=[risk_colors[t] for t in risk_df["risk_tier"]],
        edgecolor=COLORS["border"], linewidth=0.5
    )
    ax2.set_xlabel("Risk Score")
    ax2.set_title("Zone Risk Assessment", fontsize=11, fontweight="bold")
    for bar, tier in zip(bars, risk_df["risk_tier"]):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 tier, va="center", fontsize=7, color=risk_colors[tier])

    # 3. All zones — average daily load profile
    ax3 = fig.add_subplot(gs[1, :2])
    style_ax(ax3)
    zone_colors = [COLORS["blue"], COLORS["green"], COLORS["orange"],
                   COLORS["purple"], COLORS["yellow"]]
    for i, (z, fdf2) in enumerate(forecasts.items()):
        fdf2 = fdf2.copy()
        fdf2["hour"] = fdf2["timestamp"].dt.hour
        hourly_avg = fdf2.groupby("hour")["total_kwh"].mean()
        ax3.plot(hourly_avg.index, hourly_avg.values,
                 label=z, color=zone_colors[i % len(zone_colors)], lw=1.5)
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("Avg Total kWh")
    ax3.set_title("Daily Load Profiles by Zone", fontsize=11, fontweight="bold")
    ax3.legend(facecolor=COLORS["panel"], edgecolor=COLORS["border"],
               labelcolor=COLORS["text"], fontsize=7)

    # 4. Model metrics table
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(COLORS["panel"])
    ax4.axis("off")
    ax4.set_title("Model Performance", fontsize=11, fontweight="bold", color=COLORS["text"])

    table_data = []
    for z, fdf3 in forecasts.items():
        mae_m = fdf3["model_mae"].mean()
        mae_b = fdf3["baseline_mae"].mean()
        improvement = (mae_b - mae_m) / (mae_b + 1e-8) * 100
        table_data.append([z[:10], f"{mae_m:.2f}", f"{mae_b:.2f}", f"{improvement:.0f}%"])

    col_labels = ["Zone", "MAE\n(Model)", "MAE\n(Baseline)", "Improve"]
    tbl = ax4.table(
        cellText=table_data, colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0, 0.05, 1, 0.9]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(COLORS["bg"] if r == 0 else COLORS["panel"])
        cell.set_text_props(color=COLORS["text"] if r > 0 else COLORS["blue"])
        cell.set_edgecolor(COLORS["border"])

    fig.suptitle("BESCOM Smart Meter Intelligence — Demand Forecasting Dashboard",
                 fontsize=14, fontweight="bold", color=COLORS["text"], y=0.98)

    path = "outputs/plots/demand_forecast.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_anomaly_results(anomaly_df):
    """Comprehensive anomaly detection visualization."""
    fig = plt.figure(figsize=(18, 12), facecolor=COLORS["bg"])
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    flagged = anomaly_df[anomaly_df["detected_anomaly"]]
    normal = anomaly_df[~anomaly_df["detected_anomaly"]]

    # 1. Scatter: mean consumption vs CV colored by type
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1)
    type_colors = {
        "normal": COLORS["green"],
        "theft_gradual": COLORS["red"],
        "theft_sudden_drop": COLORS["orange"],
        "tamper_spike": COLORS["purple"],
        "dead_meter": COLORS["yellow"],
        "peer_deviation": COLORS["blue"],
        "statistical_anomaly": COLORS["muted"],
    }
    for dtype, grp in anomaly_df.groupby("detected_type"):
        ax1.scatter(grp["mean_consumption"], grp["cv"],
                    c=type_colors.get(dtype, COLORS["muted"]),
                    label=dtype, alpha=0.7, s=40, edgecolors="none")
    ax1.set_xlabel("Mean Consumption (kWh)")
    ax1.set_ylabel("Coefficient of Variation")
    ax1.set_title("Anomaly Detection — Consumption vs Variability", fontsize=11, fontweight="bold")
    ax1.legend(facecolor=COLORS["panel"], edgecolor=COLORS["border"],
               labelcolor=COLORS["text"], fontsize=7, markerscale=1.2)

    # 2. Confidence score distribution
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2)
    if len(flagged) > 0:
        ax2.hist(flagged["confidence_score"], bins=15, color=COLORS["red"],
                 alpha=0.8, edgecolor=COLORS["border"], linewidth=0.5)
    ax2.set_xlabel("Confidence Score")
    ax2.set_ylabel("Count")
    ax2.set_title("Flagged Meters — Confidence Distribution", fontsize=11, fontweight="bold")

    # 3. Anomaly breakdown by zone
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3)
    zone_counts = flagged.groupby("zone").size().sort_values(ascending=True)
    ax3.barh(zone_counts.index, zone_counts.values, color=COLORS["red"],
             edgecolor=COLORS["border"], linewidth=0.5)
    ax3.set_xlabel("Flagged Meters")
    ax3.set_title("Anomalies by Zone", fontsize=11, fontweight="bold")

    # 4. Detection type breakdown
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4)
    type_counts = flagged["detected_type"].value_counts()
    wedges, texts, autotexts = ax4.pie(
        type_counts.values,
        labels=type_counts.index,
        autopct="%1.0f%%",
        colors=[type_colors.get(t, COLORS["muted"]) for t in type_counts.index],
        textprops={"color": COLORS["text"], "fontsize": 7},
        pctdistance=0.75,
    )
    ax4.set_title("Anomaly Types Detected", fontsize=11, fontweight="bold")

    # 5. Confusion matrix-style stats
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(COLORS["panel"])
    ax5.axis("off")
    ax5.set_title("Detection Summary", fontsize=11, fontweight="bold", color=COLORS["text"])

    tp = ((anomaly_df["is_anomaly_true"] == 1) & (anomaly_df["detected_anomaly"])).sum()
    fp = ((anomaly_df["is_anomaly_true"] == 0) & (anomaly_df["detected_anomaly"])).sum()
    fn = ((anomaly_df["is_anomaly_true"] == 1) & (~anomaly_df["detected_anomaly"])).sum()
    tn = ((anomaly_df["is_anomaly_true"] == 0) & (~anomaly_df["detected_anomaly"])).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    summary_text = [
        f"Total Meters:    {len(anomaly_df)}",
        f"True Anomalies:  {anomaly_df['is_anomaly_true'].sum()}",
        f"Detected:        {anomaly_df['detected_anomaly'].sum()}",
        "",
        f"True Positives:  {tp}",
        f"False Positives: {fp}",
        f"False Negatives: {fn}",
        f"True Negatives:  {tn}",
        "",
        f"Precision:  {prec:.2%}",
        f"Recall:     {rec:.2%}",
        f"F1 Score:   {f1:.2%}",
    ]
    for i, line in enumerate(summary_text):
        color = COLORS["text"]
        if "False Positives" in line:
            color = COLORS["yellow"]
        elif "True Positives" in line or "Recall" in line or "F1" in line:
            color = COLORS["green"]
        ax5.text(0.05, 0.92 - i * 0.07, line, transform=ax5.transAxes,
                 fontsize=9, color=color, fontfamily="monospace")

    fig.suptitle("BESCOM Smart Meter Intelligence — Anomaly & Theft Detection Dashboard",
                 fontsize=14, fontweight="bold", color=COLORS["text"], y=0.98)

    path = "outputs/plots/anomaly_detection.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_meter_deep_dive(raw_df, anomaly_df, meter_id=None):
    """Show a single meter's time series with anomaly overlay."""
    if meter_id is None:
        flagged_meters = anomaly_df[anomaly_df["is_anomaly_true"] == 1]["meter_id"].values
        if len(flagged_meters) == 0:
            return
        meter_id = flagged_meters[0]

    mdf = raw_df[raw_df["meter_id"] == meter_id].sort_values("timestamp")
    meta = anomaly_df[anomaly_df["meter_id"] == meter_id].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=COLORS["bg"])

    for ax in axes:
        style_ax(ax)

    # Time series
    axes[0].fill_between(mdf["timestamp"], mdf["consumption_kwh"],
                         where=~mdf["is_anomaly"], alpha=0.3, color=COLORS["blue"], label="Normal")
    axes[0].fill_between(mdf["timestamp"], mdf["consumption_kwh"],
                         where=mdf["is_anomaly"], alpha=0.5, color=COLORS["red"], label="Anomalous period")
    axes[0].plot(mdf["timestamp"], mdf["consumption_kwh"], lw=0.8, color=COLORS["blue"], alpha=0.7)
    axes[0].set_title(f"Meter: {meter_id} | Zone: {meta['zone']} | Type: {meta['detected_type']}", fontsize=11)
    axes[0].set_ylabel("Consumption (kWh / 15min)")
    axes[0].legend(facecolor=COLORS["panel"], edgecolor=COLORS["border"],
                   labelcolor=COLORS["text"], fontsize=8)

    # Daily totals
    daily = mdf.set_index("timestamp").resample("D")["consumption_kwh"].sum()
    daily_anom = mdf[mdf["is_anomaly"]].set_index("timestamp").resample("D")["consumption_kwh"].sum()
    axes[1].bar(daily.index, daily.values, color=COLORS["blue"], alpha=0.6, label="Daily Total", width=0.8)
    if len(daily_anom) > 0:
        axes[1].bar(daily_anom.index, daily_anom.values, color=COLORS["red"],
                    alpha=0.8, label="Anomalous Days", width=0.8)
    axes[1].set_title("Daily Consumption Totals", fontsize=10)
    axes[1].set_ylabel("kWh / Day")
    axes[1].set_xlabel("Date")
    axes[1].legend(facecolor=COLORS["panel"], edgecolor=COLORS["border"],
                   labelcolor=COLORS["text"], fontsize=8)

    # Annotation box
    info = (f"Confidence: {meta['confidence_score']:.0f}/100\n"
            f"Reason: {meta['rule_reasons'][:80]}...")
    axes[0].text(0.01, 0.96, info, transform=axes[0].transAxes,
                 fontsize=7, color=COLORS["yellow"], va="top",
                 bbox=dict(boxstyle="round", fc=COLORS["bg"], ec=COLORS["yellow"], alpha=0.8))

    fig.suptitle("Meter Deep Dive — Anomaly Analysis", fontsize=13,
                 fontweight="bold", color=COLORS["text"])

    path = f"outputs/plots/meter_deepdive_{meter_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"Saved: {path}")
    return path


def generate_inspection_report(anomaly_df, risk_df):
    """Generate actionable inspection report CSV."""
    flagged = anomaly_df[anomaly_df["detected_anomaly"]].copy()
    flagged["priority"] = pd.cut(
        flagged["confidence_score"],
        bins=[0, 40, 65, 100],
        labels=["LOW", "MEDIUM", "HIGH"]
    )
    report = flagged[[
        "meter_id", "zone", "detected_type", "confidence_score", "priority",
        "mean_consumption", "recent_hist_ratio", "zero_ratio",
        "peer_deviation_score", "rule_reasons"
    ]].sort_values(["priority", "confidence_score"], ascending=[True, False])

    report.to_csv("outputs/inspection_report.csv", index=False)
    print(f"Saved: outputs/inspection_report.csv ({len(report)} meters flagged)")
    return report


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def run_pipeline():
    print("=" * 60)
    print("  BESCOM Smart Meter Intelligence System")
    print("  AI for Bharat Hackathon 2026")
    print("=" * 60)

    # ── Step 1: Generate Data ──
    print("\n[1/5] Generating synthetic smart meter data...")
    raw_df = generate_full_dataset(days=60)
    raw_df.to_csv("data/smart_meter_data.csv", index=False)
    print(f"  Records: {len(raw_df):,} | Meters: {raw_df['meter_id'].nunique()} | Zones: {raw_df['zone'].nunique()}")

    # ── Step 2: Zone-Level Demand Forecasting ──
    print("\n[2/5] Preparing zone-level hourly aggregations...")
    zone_hourly = prepare_zone_hourly(raw_df)

    print("\n[3/5] Training demand forecasting models (GradientBoosting)...")
    forecaster = DemandForecaster()
    forecaster.train(zone_hourly)
    forecasts = forecaster.predict(zone_hourly)

    risk_df = compute_zone_risk(forecasts)
    risk_df.to_csv("outputs/zone_risk.csv", index=False)
    print("\n  Zone Risk Summary:")
    print(risk_df[["zone", "risk_tier", "risk_score", "peak_load_kwh"]].to_string(index=False))

    # ── Step 3: Anomaly Detection ──
    print("\n[4/5] Computing meter features for anomaly detection...")
    meter_feat = compute_meter_features(raw_df)

    detector = AnomalyDetector(contamination=0.15)
    anomaly_results = detector.fit_predict(meter_feat)
    anomaly_results.to_csv("outputs/anomaly_results.csv", index=False)

    inspection_report = generate_inspection_report(anomaly_results, risk_df)

    # ── Step 4: Visualizations ──
    print("\n[5/5] Generating visualizations...")
    plot_demand_forecast(forecasts, risk_df)
    plot_anomaly_results(anomaly_results)

    # Deep dive for a theft meter and a dead meter
    for atype in ["theft_gradual", "dead_meter", "tamper_spike"]:
        candidates = anomaly_results[anomaly_results["anomaly_type_true"] == atype]["meter_id"].values
        if len(candidates) > 0:
            plot_meter_deep_dive(raw_df, anomaly_results, candidates[0])

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Outputs:")
    print(f"    data/smart_meter_data.csv           — synthetic meter data")
    print(f"    outputs/zone_risk.csv               — zone risk assessment")
    print(f"    outputs/anomaly_results.csv         — per-meter anomaly analysis")
    print(f"    outputs/inspection_report.csv       — prioritized inspection list")
    print(f"    outputs/plots/demand_forecast.png   — forecasting dashboard")
    print(f"    outputs/plots/anomaly_detection.png — anomaly dashboard")
    print(f"\n  Model Metrics:")
    for zone, m in forecaster.metrics.items():
        print(f"    {zone:15s} — MAE:{m['MAE']:.2f} RMSE:{m['RMSE']:.2f} R²:{m['R2']:.3f}")
    print(f"\n  Anomaly Detection:")
    for k, v in detector.metrics.items():
        print(f"    {k:12s}: {v}")
    print()

    return forecaster, detector, forecasts, anomaly_results, risk_df


if __name__ == "__main__":
    run_pipeline()
