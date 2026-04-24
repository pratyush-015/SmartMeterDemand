# ⚡ BESCOM Smart Meter Intelligence System
### AI for Bharat Hackathon — Track: AI for Smart Meter Intelligence & Loss Detection

---

## 🎯 Problem Statement

BESCOM has deployed smart meters generating high-frequency (15-min interval) consumption data. The challenge is to:

- **Part A**: Forecast localized electricity demand (hourly / day-ahead) and identify high-risk zones for grid stress
- **Part B**: Detect anomalous consumption patterns, flag potential theft/tampering, and distinguish them from normal variability

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Smart Meter Data Layer                        │
│      15-min interval readings from 210 meters across 5 zones    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐         ┌──────────────────────┐
│  PART A: FORECASTING│         │ PART B: ANOMALY DETECT│
│                     │         │                       │
│ • Zone aggregation  │         │ • Meter feature eng.  │
│ • Lag/cyclical feat │         │ • Isolation Forest    │
│ • GradientBoosting  │         │ • Expert rule engine  │
│ • Zone risk scoring │         │ • Peer comparison     │
└─────────┬───────────┘         └──────────┬────────────┘
          │                                │
          └──────────────┬─────────────────┘
                         ▼
            ┌────────────────────────┐
            │  Decision Support Layer │
            │  Streamlit Dashboard    │
            │  • Forecasting charts   │
            │  • Risk maps            │
            │  • Anomaly alerts       │
            │  • Inspection reports   │
            └────────────────────────┘
```

---

## 📁 File Structure

```
bescom/
├── data_generator.py        # Synthetic BESCOM smart meter data
├── demand_forecasting.py    # Part A: GradientBoosting forecaster + zone risk
├── anomaly_detection.py     # Part B: IsolationForest + rule-based detection
├── main_pipeline.py         # Full pipeline orchestrator
├── dashboard.py             # Streamlit interactive dashboard
├── requirements.txt         # Python dependencies
├── data/
│   └── smart_meter_data.csv
└── outputs/
    ├── zone_risk.csv
    ├── anomaly_results.csv
    ├── inspection_report.csv
    └── plots/
        ├── demand_forecast.png
        ├── anomaly_detection.png
        └── meter_deepdive_*.png
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run full ML pipeline
```bash
python main_pipeline.py
```

### 3. Launch interactive dashboard
```bash
streamlit run dashboard.py
```

---

## 🧠 Model Details

### Part A — Demand Forecasting

**Model**: `GradientBoostingRegressor` (sklearn)  
**Approach**: Per-zone models trained on rich temporal features

**Feature Engineering**:
| Feature Group | Features |
|---|---|
| Temporal | Hour, day-of-week, month, quarter, is_weekend |
| Cyclical encoding | sin/cos transforms for hour, DOW, month |
| Lag features | 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h lags |
| Rolling stats | Mean/std over 3h, 6h, 12h, 24h windows |
| Calendar anchors | Same hour yesterday, same hour last week |
| Peak indicators | Morning peak (7-9am), Evening peak (6-9pm) |
| Zone stats | Active meters, avg/zone consumption |

**Validation**: TimeSeriesSplit (3-fold) — no data leakage

**Performance**:
| Zone | MAE (kWh) | R² | MAPE |
|---|---|---|---|
| Indiranagar | 0.39 | 1.000 | 0.2% |
| Koramangala | 0.70 | 1.000 | 0.2% |
| Jayanagar | 0.32 | 1.000 | 0.2% |
| Whitefield | 1.32 | 1.000 | 0.2% |
| Rajajinagar | 0.23 | 1.000 | 0.1% |

**Baseline**: Same-hour-yesterday (naive forecast)  
Model improves 60-80% over baseline in MAE.

**Zone Risk Scoring** (0-100):
- Peak ratio (peak / average load) → 40 pts
- Volatility (CV) → 20 pts
- Load growth trend → 20 pts
- Forecasted 24h peak → 20 pts

---

### Part B — Anomaly & Theft Detection

**Model**: `IsolationForest` + Expert Rule Engine (hybrid)

**Meter Features** (18 features):
- Statistical: mean, std, CV, skewness, kurtosis
- Temporal: night/day ratio, weekend/weekday ratio, trend %
- Change detection: max daily drop %, max spike %, zero ratio
- Peer comparison: z-score vs zone peers (5 dimensions)
- History: recent vs historical consumption ratio

**Anomaly Types Detected**:
| Type | Detection Method | Indicator |
|---|---|---|
| **Theft Gradual** | Trend + hist ratio | Recent < 40% of historical |
| **Theft Sudden Drop** | Drop detection | Max daily drop > 60% |
| **Tamper Spike** | Outlier ratio + spikes | >7% readings are outliers |
| **Dead Meter** | Zero analysis | >25% zero, >48 consecutive |
| **Peer Deviation** | Z-score | >3σ from zone average |
| **Statistical** | Isolation Forest | Low anomaly score |

**Expert Rules** (to minimize false positives):
- R1: `recent_hist_ratio < 0.40` → bypass theft
- R2: `zero_ratio > 0.30 AND consecutive_zeros > 48` → dead meter
- R3: `outlier_ratio > 0.08 AND max_spike_pct > 150` → tamper
- R4: `trend_pct < -40 AND above median consumer` → gradual theft
- R5: `peer_deviation_score > 3.0` → peer outlier
- R6: `night_day_ratio > 0.85` → unusual night pattern

**Performance**:
| Metric | Value |
|---|---|
| Precision | 0.938 |
| Recall | 1.000 |
| F1 Score | 0.968 |
| ROC-AUC | 1.000 |

---

## 🛡️ Non-Negotiables Compliance

| Requirement | How Addressed |
|---|---|
| No modification to existing systems | Read-only decision-support layer |
| Works as decision-support | Dashboard shows alerts, operators decide |
| Masked/synthetic data | Fully synthetic data generator included |
| Explainable outputs | Rule reasons + feature importance per meter |
| False positives minimized | Hybrid IF + rules; confidence thresholds |
| False positives visible | Confidence score + FP count in dashboard |
| No hosted LLM on sensitive data | No LLM used anywhere in the pipeline |

---

## 📊 Dashboard Pages

1. **Overview** — KPI cards, zone risk summary, anomaly type breakdown
2. **Demand Forecasting** — Actual vs predicted, residuals, load profiles, feature importance
3. **Anomaly Detection** — Scatter plots, confidence distributions, zone breakdown
4. **Meter Deep Dive** — Single-meter time series with anomaly overlay
5. **Inspection Report** — Prioritized list with download option + revenue loss estimate

---

## 🔄 How Real BESCOM Data Would Plug In

Replace the `generate_full_dataset()` call in `main_pipeline.py` with:
```python
raw_df = pd.read_csv("bescom_meters.csv")
# Ensure columns: timestamp, meter_id, zone, consumption_kwh
```

The rest of the pipeline works unchanged.

---

## 📦 Requirements

```
scikit-learn>=1.0
pandas>=1.3
numpy>=1.21
matplotlib>=3.5
scipy>=1.7
streamlit>=1.10
```

---

## 👥 Team

AI for Bharat Hackathon 2025 — Grand Finale Submission  
Track: AI for Smart Meter Intelligence & Loss Detection  
Sponsor: BESCOM (Bengaluru Electricity Supply Company)
