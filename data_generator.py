"""
BESCOM Smart Meter Synthetic Data Generator
Generates realistic 15-minute interval smart meter data for multiple zones/meters
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# Zone & meter configuration
ZONES = {
    "Indiranagar": {"meters": 40, "base_load": 2.5, "peak_multiplier": 2.8, "commercial_ratio": 0.3},
    "Koramangala":  {"meters": 45, "base_load": 3.0, "peak_multiplier": 3.2, "commercial_ratio": 0.5},
    "Jayanagar":    {"meters": 35, "base_load": 2.2, "peak_multiplier": 2.5, "commercial_ratio": 0.2},
    "Whitefield":   {"meters": 60, "base_load": 3.5, "peak_multiplier": 3.8, "commercial_ratio": 0.6},
    "Rajajinagar":  {"meters": 30, "base_load": 2.0, "peak_multiplier": 2.3, "commercial_ratio": 0.2},
}

ANOMALY_TYPES = [
    "theft_gradual",      # slow consistent undercounting
    "theft_sudden_drop",  # abrupt drop suggesting bypass
    "tamper_spike",       # erratic high spikes
    "dead_meter",         # consumption stuck at zero
    "peer_deviation",     # outlier vs similar meters
]


def hourly_pattern(hour, is_commercial):
    """Generate consumption multiplier based on hour of day."""
    if is_commercial:
        # Business hours peak
        if 9 <= hour <= 18:
            return 1.5 + 0.8 * np.sin(np.pi * (hour - 9) / 9)
        elif 19 <= hour <= 22:
            return 0.8
        else:
            return 0.2
    else:
        # Residential: morning + evening peaks
        morning = 0.6 * np.exp(-0.5 * ((hour - 7.5) / 1.5) ** 2)
        evening = 1.0 * np.exp(-0.5 * ((hour - 19.5) / 2.0) ** 2)
        night_base = 0.15
        return night_base + morning + evening


def seasonal_factor(month):
    """Karnataka climate: hot March-May, moderate monsoon June-Sep."""
    seasonal = {1: 0.85, 2: 0.90, 3: 1.10, 4: 1.25, 5: 1.30,
                6: 1.05, 7: 0.95, 8: 0.90, 9: 0.95, 10: 1.0, 11: 0.90, 12: 0.85}
    return seasonal[month]


def generate_meter_data(meter_id, zone_name, zone_cfg, start_date, days=90, anomaly_type=None):
    """Generate 15-min interval data for a single meter."""
    freq = "15min"
    timestamps = pd.date_range(start=start_date, periods=days * 96, freq=freq)
    is_commercial = np.random.random() < zone_cfg["commercial_ratio"]
    base = zone_cfg["base_load"] * (0.7 + 0.6 * np.random.random())

    records = []
    for ts in timestamps:
        hour = ts.hour + ts.minute / 60
        h_factor = hourly_pattern(ts.hour, is_commercial)
        s_factor = seasonal_factor(ts.month)
        weekday_factor = 0.85 if ts.weekday() >= 5 and is_commercial else 1.0

        # Base consumption (kWh per 15 min)
        consumption = base * h_factor * s_factor * weekday_factor
        # Add realistic noise
        consumption += np.random.normal(0, consumption * 0.08)
        consumption = max(0.0, consumption)

        records.append({
            "timestamp": ts,
            "meter_id": meter_id,
            "zone": zone_name,
            "consumption_kwh": round(consumption, 4),
            "is_commercial": is_commercial,
            "anomaly_type": None,
            "is_anomaly": False,
        })

    df = pd.DataFrame(records)

    # Inject anomalies
    if anomaly_type:
        df = inject_anomaly(df, anomaly_type)

    return df

def generate_meter_data_vec(meter_id, zone_name, zone_cfg, start_date, days=90, anomaly_type=None):
    freq = "15min"
    periods = days * 96
    timestamps = pd.date_range(start=start_date, periods=periods, freq=freq)

    is_commercial = np.random.random() < zone_cfg["commercial_ratio"]
    base = zone_cfg["base_load"] * (0.7 + 0.6*np.random.random())

    hours = timestamps.hour + timestamps.minute / 60.0
    months = timestamps.month
    weekdays = timestamps.weekday

    if is_commercial:
        h_factors = np.where((hours >= 9) & (hours <= 18),
                             1.5 + 0.8*np.sin(np.pi * (hours - 9)/9),
                             np.where((hours >= 19) & (hours <= 22), 0.8, 0.2))
    else:
        morning = 0.6 * np.exp(-0.5 * ((hours - 7.5) / 1.5) ** 2)
        evening = 1.0 * np.exp(-0.5 * ((hours - 19.5) / 2.0) ** 2)
        h_factors = 0.15 + morning + evening

    s_map = np.array([seasonal_factor(m) for m in range(1, 13)])
    s_factors = s_map[months - 1]
    
    weekday_factors = np.where((weekdays >= 5) & is_commercial, 0.85, 1.0)

    consumption = base * h_factors * s_factors * weekday_factors

    consumption += np.random.normal(0, consumption * 0.08, size=periods)
    consumption = np.maximum(0.0, consumption)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "meter_id": meter_id,
        "zone": zone_name,
        "consumption_kwh": np.round(consumption, 4),
        "is_commercial": is_commercial,
        "anomaly_type": None,
        "is_anomaly": False,
    })

    if anomaly_type:
        df = inject_anomaly(df, anomaly_type)

    return df


def inject_anomaly(df, anomaly_type):
    """Inject realistic anomaly patterns into meter data."""
    n = len(df)

    if anomaly_type == "theft_gradual":
        # Last 30 days: gradual 40-60% reduction
        cutoff = int(n * 0.67)
        reduction = np.linspace(1.0, 0.4, n - cutoff)
        df.loc[df.index[cutoff:], "consumption_kwh"] *= reduction
        df.loc[df.index[cutoff:], "is_anomaly"] = True
        df.loc[df.index[cutoff:], "anomaly_type"] = "theft_gradual"

    elif anomaly_type == "theft_sudden_drop":
        # Sudden 70% drop from day 45 onwards
        cutoff = int(n * 0.5)
        df.loc[df.index[cutoff:], "consumption_kwh"] *= 0.28
        df.loc[df.index[cutoff:], "is_anomaly"] = True
        df.loc[df.index[cutoff:], "anomaly_type"] = "theft_sudden_drop"

    elif anomaly_type == "tamper_spike":
        # Random high spikes (meter tampering causing overcounting)
        spike_idx = np.random.choice(df.index[n // 2:], size=int(n * 0.03), replace=False)
        df.loc[spike_idx, "consumption_kwh"] *= np.random.uniform(4, 8, len(spike_idx))
        df.loc[spike_idx, "is_anomaly"] = True
        df.loc[spike_idx, "anomaly_type"] = "tamper_spike"

    elif anomaly_type == "dead_meter":
        # Meter stuck at near zero for 3 weeks
        cutoff = int(n * 0.6)
        end = int(n * 0.8)
        df.loc[df.index[cutoff:end], "consumption_kwh"] = np.random.uniform(0, 0.01, end - cutoff)
        df.loc[df.index[cutoff:end], "is_anomaly"] = True
        df.loc[df.index[cutoff:end], "anomaly_type"] = "dead_meter"

    elif anomaly_type == "peer_deviation":
        # Consistently 3x higher than peers (undisclosed commercial use)
        cutoff = int(n * 0.4)
        df.loc[df.index[cutoff:], "consumption_kwh"] *= np.random.uniform(2.8, 3.5)
        df.loc[df.index[cutoff:], "is_anomaly"] = True
        df.loc[df.index[cutoff:], "anomaly_type"] = "peer_deviation"

    return df


def generate_full_dataset(start_date="2024-01-01", days=90):
    """Generate complete dataset for all zones and meters."""
    all_dfs = []
    meter_counter = 1000

    for zone_name, zone_cfg in ZONES.items():
        n_meters = zone_cfg["meters"]
        # ~15% meters have anomalies
        n_anomalous = max(2, int(n_meters * 0.15))
        anomaly_assignments = (
            [None] * (n_meters - n_anomalous) +
            list(np.random.choice(ANOMALY_TYPES, n_anomalous))
        )
        np.random.shuffle(anomaly_assignments)

        for i, atype in enumerate(anomaly_assignments):
            meter_id = f"MTR_{zone_name[:3].upper()}_{meter_counter}"
            meter_counter += 1
            df = generate_meter_data(
                meter_id, zone_name, zone_cfg,
                start_date=start_date, days=days, anomaly_type=atype
            )
            all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.sort_values(["timestamp", "zone", "meter_id"]).reset_index(drop=True)
    return full_df


if __name__ == "__main__":
    print("Generating synthetic BESCOM smart meter data...")
    df = generate_full_dataset()
    df.to_csv("data/smart_meter_data.csv", index=False)
    print(f"Generated {len(df):,} records | {df['meter_id'].nunique()} meters | {df['zone'].nunique()} zones")
    print(f"Anomalous readings: {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean()*100:.1f}%)")
    print(f"Anomalous meters: {df[df['is_anomaly']]['meter_id'].nunique()}")
    print(df.head())
