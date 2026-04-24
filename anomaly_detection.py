"""
Part B: Anomaly & Theft Detection
- Isolation Forest for statistical anomaly detection
- Rule-based expert system for theft-specific patterns
- Peer-comparison scoring
- Explainable flagging with confidence scores
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# Meter-Level Feature Engineering
# ─────────────────────────────────────────

def compute_meter_features(df_raw):
    """Compute rich per-meter statistical features for anomaly detection."""
    meter_features = []

    for meter_id, mdf in df_raw.groupby("meter_id"):
        mdf = mdf.sort_values("timestamp")
        consumption = mdf["consumption_kwh"].values

        if len(consumption) < 96:  # need at least 1 day
            continue

        # ── Basic statistics ──
        mean_c = np.mean(consumption)
        std_c = np.std(consumption)
        median_c = np.median(consumption)
        cv = std_c / (mean_c + 1e-8)  # coefficient of variation
        skewness = _skewness(consumption)
        kurtosis = _kurtosis(consumption)

        # ── Consumption ratio patterns ──
        # Night (22:00-05:00) vs day ratio
        mdf_ts = mdf.copy()
        mdf_ts["hour"] = mdf_ts["timestamp"].dt.hour
        night_mask = (mdf_ts["hour"] >= 22) | (mdf_ts["hour"] <= 5)
        night_mean = mdf_ts[night_mask]["consumption_kwh"].mean()
        day_mean = mdf_ts[~night_mask]["consumption_kwh"].mean()
        night_day_ratio = night_mean / (day_mean + 1e-8)

        # Weekend vs weekday
        mdf_ts["is_weekend"] = mdf_ts["timestamp"].dt.dayofweek >= 5
        wend_mean = mdf_ts[mdf_ts["is_weekend"]]["consumption_kwh"].mean()
        wday_mean = mdf_ts[~mdf_ts["is_weekend"]]["consumption_kwh"].mean()
        weekend_weekday_ratio = wend_mean / (wday_mean + 1e-8)

        # ── Temporal trend ──
        x = np.arange(len(consumption))
        slope = np.polyfit(x, consumption, 1)[0]
        trend_pct = slope * len(consumption) / (mean_c + 1e-8) * 100

        # ── Sudden change detection ──
        daily_totals = mdf.set_index("timestamp").resample("D")["consumption_kwh"].sum().values
        if len(daily_totals) > 7:
            daily_changes = np.diff(daily_totals)
            max_drop_pct = np.min(daily_changes) / (np.mean(daily_totals[:7]) + 1e-8) * 100
            max_spike_pct = np.max(daily_changes) / (np.mean(daily_totals[:7]) + 1e-8) * 100
        else:
            max_drop_pct = 0
            max_spike_pct = 0

        # ── Zero / near-zero readings ──
        zero_ratio = np.mean(consumption < 0.05)
        consecutive_zeros = _max_consecutive(consumption < 0.05)

        # ── Outlier ratio ──
        q1, q3 = np.percentile(consumption, [25, 75])
        iqr = q3 - q1
        outlier_ratio = np.mean(
            (consumption < q1 - 3 * iqr) | (consumption > q3 + 3 * iqr)
        )

        # ── Recent vs historical ratio (last 30 days vs first 30 days) ──
        n96 = 96 * 30  # 30 days of 15-min readings
        hist_mean = consumption[:n96].mean() if len(consumption) >= n96 * 2 else mean_c
        recent_mean = consumption[-n96:].mean() if len(consumption) >= n96 else mean_c
        recent_hist_ratio = recent_mean / (hist_mean + 1e-8)

        # ── Ground truth ──
        is_anomaly = mdf["is_anomaly"].any()
        anomaly_type = mdf[mdf["is_anomaly"]]["anomaly_type"].mode()[0] if is_anomaly else "normal"
        zone = mdf["zone"].iloc[0]

        meter_features.append({
            "meter_id": meter_id,
            "zone": zone,
            "mean_consumption": mean_c,
            "std_consumption": std_c,
            "median_consumption": median_c,
            "cv": cv,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "night_day_ratio": night_day_ratio,
            "weekend_weekday_ratio": weekend_weekday_ratio,
            "trend_pct": trend_pct,
            "max_drop_pct": max_drop_pct,
            "max_spike_pct": max_spike_pct,
            "zero_ratio": zero_ratio,
            "consecutive_zeros": consecutive_zeros,
            "outlier_ratio": outlier_ratio,
            "recent_hist_ratio": recent_hist_ratio,
            "is_anomaly_true": int(is_anomaly),
            "anomaly_type_true": anomaly_type,
        })

    return pd.DataFrame(meter_features)


def _skewness(x):
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-10:
        return 0
    return np.mean(((x - m) / s) ** 3)


def _kurtosis(x):
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-10:
        return 0
    return np.mean(((x - m) / s) ** 4) - 3


def _max_consecutive(bool_arr):
    """Find max run of True in boolean array."""
    max_run = 0
    cur_run = 0
    for v in bool_arr:
        if v:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


# ─────────────────────────────────────────
# Peer Comparison
# ─────────────────────────────────────────

def add_peer_features(meter_feat_df):
    """Add z-score deviation from zone peers."""
    df = meter_feat_df.copy()
    stat_cols = ["mean_consumption", "cv", "zero_ratio", "night_day_ratio",
                 "trend_pct", "recent_hist_ratio"]

    for col in stat_cols:
        zone_mean = df.groupby("zone")[col].transform("mean")
        zone_std = df.groupby("zone")[col].transform("std").replace(0, 1e-8)
        df[f"peer_z_{col}"] = (df[col] - zone_mean) / zone_std

    df["peer_deviation_score"] = df[[f"peer_z_{c}" for c in stat_cols]].abs().mean(axis=1)
    return df


# ─────────────────────────────────────────
# Anomaly Detection Models
# ─────────────────────────────────────────

DETECTION_FEATURES = [
    "cv", "skewness", "kurtosis", "zero_ratio", "consecutive_zeros",
    "outlier_ratio", "trend_pct", "max_drop_pct", "max_spike_pct",
    "night_day_ratio", "weekend_weekday_ratio", "recent_hist_ratio",
    "peer_z_mean_consumption", "peer_z_cv", "peer_z_zero_ratio",
    "peer_z_trend_pct", "peer_z_recent_hist_ratio",
    "peer_deviation_score",
]


class AnomalyDetector:
    def __init__(self, contamination=0.15):
        self.contamination = contamination
        self.isoforest = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.metrics = {}

    def fit_predict(self, meter_feat_df):
        """Fit and predict anomalies using Isolation Forest + rules."""
        df = meter_feat_df.copy()
        df = add_peer_features(df)

        X = df[DETECTION_FEATURES].fillna(0).values
        X_s = self.scaler.fit_transform(X)

        # Isolation Forest scores (-1 = anomaly, 1 = normal)
        df["if_label"] = self.isoforest.fit_predict(X_s)
        df["if_score"] = self.isoforest.decision_function(X_s)  # lower = more anomalous
        df["if_anomaly_prob"] = 1 - (df["if_score"] - df["if_score"].min()) / \
                                    (df["if_score"].max() - df["if_score"].min() + 1e-8)

        # Rule-based detection layer
        df = self._apply_rules(df)

        # Combined flag
        df["detected_anomaly"] = (df["if_label"] == -1) | df["rule_flag"]

        # Anomaly type classification
        df["detected_type"] = df.apply(self._classify_type, axis=1)

        # Confidence score (0-100)
        df["confidence_score"] = self._compute_confidence(df)

        # Evaluation
        y_true = df["is_anomaly_true"].values
        y_pred = df["detected_anomaly"].astype(int).values
        self.metrics = {
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
            "roc_auc": round(roc_auc_score(y_true, df["if_anomaly_prob"].values), 3),
        }
        print(f"\n[Part B] Anomaly Detection Metrics:")
        print(f"  Precision : {self.metrics['precision']}")
        print(f"  Recall    : {self.metrics['recall']}")
        print(f"  F1 Score  : {self.metrics['f1']}")
        print(f"  ROC-AUC   : {self.metrics['roc_auc']}")

        return df

    def _apply_rules(self, df):
        """Expert-system rules for theft/tamper detection."""
        rules = pd.Series(False, index=df.index)

        # R1: Sudden large drop (> 60% reduction from history)
        rules |= df["recent_hist_ratio"] < 0.40

        # R2: Dead meter (many zero readings)
        rules |= (df["zero_ratio"] > 0.30) & (df["consecutive_zeros"] > 48)

        # R3: High spike outlier ratio (tampering)
        rules |= (df["outlier_ratio"] > 0.08) & (df["max_spike_pct"] > 150)

        # R4: Strong downward trend with high historical consumption
        rules |= (df["trend_pct"] < -40) & (df["mean_consumption"] > df["mean_consumption"].quantile(0.3))

        # R5: Peer deviation beyond 3 sigma
        rules |= df["peer_deviation_score"] > 3.0

        # R6: Unusual night usage (could indicate commercial bypass)
        rules |= df["night_day_ratio"] > 0.85

        df["rule_flag"] = rules
        df["rule_reasons"] = df.apply(self._explain_rules, axis=1)
        return df

    def _explain_rules(self, row):
        reasons = []
        if row["recent_hist_ratio"] < 0.40:
            reasons.append(f"Consumption dropped to {row['recent_hist_ratio']*100:.0f}% of historical")
        if (row["zero_ratio"] > 0.30) and (row["consecutive_zeros"] > 48):
            reasons.append(f"Zero readings: {row['zero_ratio']*100:.1f}% ({row['consecutive_zeros']} consecutive)")
        if (row["outlier_ratio"] > 0.08) and (row["max_spike_pct"] > 150):
            reasons.append(f"Spike anomalies: {row['outlier_ratio']*100:.1f}% readings")
        if (row["trend_pct"] < -40):
            reasons.append(f"Strong decline trend: {row['trend_pct']:.1f}%")
        if row["peer_deviation_score"] > 3.0:
            reasons.append(f"Peer deviation: {row['peer_deviation_score']:.1f}σ from zone average")
        if row["night_day_ratio"] > 0.85:
            reasons.append(f"Abnormal night usage ratio: {row['night_day_ratio']:.2f}")
        return "; ".join(reasons) if reasons else "Statistical anomaly (Isolation Forest)"

    def _classify_type(self, row):
        if not row["detected_anomaly"]:
            return "normal"
        if row["zero_ratio"] > 0.25 and row["consecutive_zeros"] > 48:
            return "dead_meter"
        if row["recent_hist_ratio"] < 0.45:
            return "theft_sudden_drop" if row["max_drop_pct"] < -40 else "theft_gradual"
        if row["outlier_ratio"] > 0.07 and row["max_spike_pct"] > 150:
            return "tamper_spike"
        if row["peer_deviation_score"] > 2.5:
            return "peer_deviation"
        return "statistical_anomaly"

    def _compute_confidence(self, df):
        """Compute confidence score 0-100 for each flagged anomaly."""
        # Normalize multiple signals
        conf = (
            df["if_anomaly_prob"] * 40 +
            (df["rule_flag"].astype(float) * 30) +
            np.clip(df["peer_deviation_score"] / 5 * 20, 0, 20) +
            np.clip(np.abs(df["trend_pct"]) / 100 * 10, 0, 10)
        )
        return conf.clip(0, 100).round(1)


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    from data_generator import generate_full_dataset

    print("Generating data...")
    raw_df = generate_full_dataset(days=60)

    print("Computing meter features...")
    meter_feat = compute_meter_features(raw_df)
    print(f"Meters: {len(meter_feat)} | Anomalous: {meter_feat['is_anomaly_true'].sum()}")

    detector = AnomalyDetector(contamination=0.15)
    results = detector.fit_predict(meter_feat)

    flagged = results[results["detected_anomaly"]].sort_values("confidence_score", ascending=False)
    print(f"\nFlagged {len(flagged)} meters for inspection:")
    print(flagged[["meter_id", "zone", "detected_type", "confidence_score", "rule_reasons"]].to_string(index=False))

    results.to_csv("outputs/anomaly_results.csv", index=False)
    print("\nSaved: outputs/anomaly_results.csv")
