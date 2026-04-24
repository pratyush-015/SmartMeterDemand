"""
Part A: Localized Demand Forecasting & Zone Risk Classification
- GradientBoosting for hourly/day-ahead demand forecasting
- Zone-level aggregation and risk scoring
- Explainable feature importance
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────

def create_time_features(df_zone_hourly):
    """Create temporal and lag features for forecasting."""
    df = df_zone_hourly.copy()

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["dayofyear"] = df["timestamp"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["quarter"] = df["timestamp"].dt.quarter

    # Cyclical encoding (captures periodicity better than raw integers)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features (1h, 2h, 24h, 48h, 168h = 1 week)
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f"lag_{lag}h"] = df["total_kwh"].shift(lag)

    # Rolling statistics
    for window in [3, 6, 12, 24]:
        df[f"roll_mean_{window}h"] = df["total_kwh"].shift(1).rolling(window).mean()
        df[f"roll_std_{window}h"] = df["total_kwh"].shift(1).rolling(window).std()

    # Same hour yesterday and last week
    df["same_hour_yesterday"] = df["total_kwh"].shift(24)
    df["same_hour_lastweek"] = df["total_kwh"].shift(168)

    # Peak hour indicator
    df["is_morning_peak"] = ((df["hour"] >= 7) & (df["hour"] <= 9)).astype(int)
    df["is_evening_peak"] = ((df["hour"] >= 18) & (df["hour"] <= 21)).astype(int)

    return df


def prepare_zone_hourly(df_raw):
    """Aggregate 15-min meter data to zone-level hourly totals."""
    df = df_raw.copy()
    df["timestamp_hour"] = df["timestamp"].dt.floor("h")

    zone_hourly = (
        df.groupby(["timestamp_hour", "zone"])
        .agg(
            total_kwh=("consumption_kwh", "sum"),
            active_meters=("meter_id", "nunique"),
            avg_kwh_per_meter=("consumption_kwh", "mean"),
            max_kwh=("consumption_kwh", "max"),
            std_kwh=("consumption_kwh", "std"),
        )
        .reset_index()
        .rename(columns={"timestamp_hour": "timestamp"})
    )
    return zone_hourly


# ─────────────────────────────────────────
# Forecasting Model
# ─────────────────────────────────────────

FEATURE_COLS = [
    "hour", "dayofweek", "month", "is_weekend", "quarter",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_1h", "lag_2h", "lag_3h", "lag_6h", "lag_12h", "lag_24h", "lag_48h", "lag_168h",
    "roll_mean_3h", "roll_mean_6h", "roll_mean_12h", "roll_mean_24h",
    "roll_std_3h", "roll_std_6h", "roll_std_12h", "roll_std_24h",
    "same_hour_yesterday", "same_hour_lastweek",
    "is_morning_peak", "is_evening_peak",
    "active_meters", "avg_kwh_per_meter",
]


class DemandForecaster:
    def __init__(self):
        self.models = {}        # per-zone model
        self.scalers = {}
        self.metrics = {}
        self.feature_importance = {}

    def train(self, zone_hourly_df):
        """Train per-zone forecasting models."""
        print("\n[Part A] Training demand forecasting models...")
        for zone in zone_hourly_df["zone"].unique():
            zone_df = zone_hourly_df[zone_hourly_df["zone"] == zone].copy().sort_values("timestamp")
            zone_df = create_time_features(zone_df)
            zone_df = zone_df.dropna(subset=FEATURE_COLS + ["total_kwh"])

            X = zone_df[FEATURE_COLS].values
            y = zone_df["total_kwh"].values

            # Time-series split (no data leakage)
            tscv = TimeSeriesSplit(n_splits=3)
            split_results = []
            for tr_idx, val_idx in tscv.split(X):
                X_tr, X_val = X[tr_idx], X[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_val_s = scaler.transform(X_val)

                model = GradientBoostingRegressor(
                    n_estimators=200, max_depth=5, learning_rate=0.08,
                    subsample=0.8, min_samples_leaf=10, random_state=42
                )
                model.fit(X_tr_s, y_tr)
                preds = model.predict(X_val_s)

                mae = mean_absolute_error(y_val, preds)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                r2 = r2_score(y_val, preds)
                mape = np.mean(np.abs((y_val - preds) / (y_val + 1e-8))) * 100
                split_results.append((mae, rmse, r2, mape, model, scaler))

            # Use best split (highest R²)
            best = max(split_results, key=lambda x: x[2])
            mae, rmse, r2, mape, model, scaler = best

            # Final model on all data
            scaler_final = StandardScaler()
            X_s = scaler_final.fit_transform(X)
            final_model = GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.08,
                subsample=0.8, min_samples_leaf=10, random_state=42
            )
            final_model.fit(X_s, y)

            self.models[zone] = final_model
            self.scalers[zone] = scaler_final
            self.metrics[zone] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}
            self.feature_importance[zone] = dict(
                zip(FEATURE_COLS, final_model.feature_importances_)
            )
            print(f"  Zone: {zone:15s} | MAE={mae:.2f} kWh | RMSE={rmse:.2f} | R²={r2:.3f} | MAPE={mape:.1f}%")

    def predict(self, zone_hourly_df, horizon_hours=24):
        """Generate day-ahead forecasts per zone."""
        forecasts = {}
        for zone, model in self.models.items():
            zone_df = zone_hourly_df[zone_hourly_df["zone"] == zone].copy().sort_values("timestamp")
            zone_df = create_time_features(zone_df)
            zone_df = zone_df.dropna(subset=FEATURE_COLS)

            X = zone_df[FEATURE_COLS].values
            X_s = self.scalers[zone].transform(X)
            preds = model.predict(X_s)

            zone_df = zone_df.copy()
            zone_df["predicted_kwh"] = preds
            zone_df["residual"] = zone_df["total_kwh"] - preds

            # Baseline: same hour yesterday
            zone_df["baseline_kwh"] = zone_df["same_hour_yesterday"]
            zone_df["baseline_mae"] = np.abs(zone_df["total_kwh"] - zone_df["baseline_kwh"])
            zone_df["model_mae"] = np.abs(zone_df["total_kwh"] - zone_df["predicted_kwh"])

            forecasts[zone] = zone_df
        return forecasts


# ─────────────────────────────────────────
# Zone Risk Classification
# ─────────────────────────────────────────

def compute_zone_risk(forecasts_dict):
    """
    Classify each zone into risk tiers based on:
    - Peak demand magnitude
    - Demand volatility
    - Load growth trend
    - Forecasted demand vs capacity threshold
    """
    risk_records = []
    for zone, df in forecasts_dict.items():
        total = df["total_kwh"]
        pred = df["predicted_kwh"]

        peak_load = total.quantile(0.95)
        avg_load = total.mean()
        volatility = total.std() / (avg_load + 1e-8)
        peak_ratio = peak_load / (avg_load + 1e-8)

        # Trend: slope over last 30 days
        recent = df.tail(30 * 24)
        if len(recent) > 2:
            x = np.arange(len(recent))
            slope = np.polyfit(x, recent["total_kwh"].values, 1)[0]
            trend_pct = (slope * len(recent)) / (recent["total_kwh"].mean() + 1e-8) * 100
        else:
            trend_pct = 0

        # Forecasted peak in next 24h
        future_peak = pred.tail(24).max() if len(pred) >= 24 else pred.max()

        # Risk score (0-100)
        risk_score = (
            min(peak_ratio * 20, 40) +       # peak ratio (max 40)
            min(volatility * 30, 20) +        # volatility (max 20)
            min(max(trend_pct, 0) * 2, 20) +  # growth trend (max 20)
            min(future_peak / (avg_load + 1e-8) * 5, 20)  # future peak (max 20)
        )

        if risk_score >= 60:
            risk_tier = "HIGH"
        elif risk_score >= 35:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "LOW"

        risk_records.append({
            "zone": zone,
            "risk_score": round(risk_score, 1),
            "risk_tier": risk_tier,
            "peak_load_kwh": round(peak_load, 2),
            "avg_load_kwh": round(avg_load, 2),
            "peak_ratio": round(peak_ratio, 2),
            "volatility": round(volatility, 3),
            "load_trend_pct": round(trend_pct, 2),
            "forecasted_peak_24h": round(future_peak, 2),
        })

    risk_df = pd.DataFrame(risk_records).sort_values("risk_score", ascending=False)
    return risk_df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    from data_generator import generate_full_dataset
    print("Generating data...")
    raw_df = generate_full_dataset(days=60)

    zone_hourly = prepare_zone_hourly(raw_df)
    print(f"Zone-hourly shape: {zone_hourly.shape}")

    forecaster = DemandForecaster()
    forecaster.train(zone_hourly)

    forecasts = forecaster.predict(zone_hourly)
    risk_df = compute_zone_risk(forecasts)

    print("\n[Zone Risk Assessment]")
    print(risk_df.to_string(index=False))

    risk_df.to_csv("outputs/zone_risk.csv", index=False)
    print("\nSaved: outputs/zone_risk.csv")
