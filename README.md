# ⚡ Smart Meter Intelligence System
Demand forecasting + theft/tamper detection on **synthetic** 15-min smart-meter data (210 meters, 5 Bengaluru zones, ~1.2M readings). Built for the BESCOM problem statement, AI for Bharat Hackathon (team HYSTREL).

**Live demo:** <PASTE STREAMLIT URL> · first load ~35 s (models train on start-up)

## Method
- **Part A:** one GradientBoostingRegressor per zone, learns the deviation from same-hour-yesterday. Day-ahead safe (all lags/rolling ≥ 24 h), cyclical time features, 3-fold `TimeSeriesSplit`, out-of-fold metrics, naive baseline = same hour yesterday. Zone risk score 0-100 (peak ratio 40, volatility 20, growth 20, forecast peak 20).
- **Part B:** 18 per-meter features + Isolation Forest + 6 expert rules (explainable reasons, fewer false positives).
- **Dashboard:** Overview, Demand Forecast, Anomaly Detection, Meter Deep Dive, Inspection Report (₹ loss estimate, CSV).

## Results (synthetic data, out-of-fold)
Forecast MAPE ≈ 2-3 %, R² ≈ 0.99, MAE 12-50 % below the naive baseline per zone. Anomaly: precision 0.938, recall 1.0, F1 0.968 (injected anomalies, so this validates the pipeline, not real-world accuracy).

## Honest notes
Data is synthetic with clean patterns and injected anomalies; real data would be noisier. An earlier version leaked a same-hour feature (R²≈1.000); removed. Decision-support only; no LLM on sensitive data.

## Run
`pip install -r requirements.txt && streamlit run dashboard.py`
