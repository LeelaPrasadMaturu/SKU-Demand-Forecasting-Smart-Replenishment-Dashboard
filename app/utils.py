# app/utils.py - helper utilities for the SKU forecasting & replenishment demo
import pandas as pd
import numpy as np
from joblib import dump, load
import os

def load_data(sales_path='data/synthetic_sales.csv', meta_path='data/sku_metadata.csv'):
    sales = pd.read_csv(sales_path, parse_dates=['date'])
    meta = pd.read_csv(meta_path)
    return sales, meta

def trailing_stats(sales_df, sku, as_of_date, window_days=30):
    sub = sales_df[(sales_df['sku_id']==sku) & (sales_df['date'] <= pd.to_datetime(as_of_date))]
    tail = sub.sort_values('date').tail(window_days)
    avg = float(tail['units_sold'].mean()) if not tail.empty else 0.0
    std = float(tail['units_sold'].std(ddof=0)) if not tail.empty else 0.0
    return avg, std

# --- Model loading and forecasting utilities ---
def load_prophet_model(sku, model_dir='../models/prophet_models'):
    """Load Prophet model for a given SKU."""
    import joblib
    model_path = os.path.join('models', 'prophet_models', f'prophet_{sku}.joblib')
    abs_path = os.path.abspath(model_path)
    print(f"[DEBUG] Looking for Prophet model at: {abs_path}")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def forecast_with_prophet(model, history_df, periods=30):
    """Forecast using a loaded Prophet model and return forecast DataFrame."""
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast.tail(periods)

def load_xgb_model(sku, model_dir='../models'):
    """Load XGBoost model for a given SKU."""
    import joblib
    model_path = os.path.join('models', f'xgb_{sku}.joblib')
    abs_path = os.path.abspath(model_path)
    print(f"[DEBUG] Looking for XGBoost model at: {abs_path}")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def forecast_with_xgb(model, df, features, periods=30):
    """Forecast using a loaded XGBoost model. Assumes df contains all required features for prediction."""
    import numpy as np
    # Use the last available row as the base for rolling prediction
    preds = []
    last_row = df.iloc[-1].copy()
    for _ in range(periods):
        X_pred = last_row[features].values.reshape(1, -1)
        y_pred = model.predict(X_pred)[0]
        preds.append(y_pred)
        # Update lags and rolling features for next step (simplified)
        for lag in [1, 7, 14, 28]:
            last_row[f'lag_{lag}'] = y_pred if lag == 1 else last_row.get(f'lag_{lag-1}', y_pred)
        last_row['rolling_mean_7'] = np.mean(preds[-7:]) if len(preds) >= 7 else np.mean(preds)
        last_row['rolling_std_14'] = np.std(preds[-14:]) if len(preds) >= 14 else np.std(preds)
        # Optionally update date, day_of_week, is_weekend, etc.
    return np.array(preds)

def compute_safety_stock(std_daily, lead_time_days, z=1.645):
    return z * std_daily * np.sqrt(lead_time_days)

def compute_reorder_point(avg_daily, lead_time_days, std_daily, z=1.645):
    safety = compute_safety_stock(std_daily, lead_time_days, z)
    return avg_daily * lead_time_days + safety

def seasonal_naive(df, horizon=30):
    last = df.sort_values('date').tail(7)['units_sold'].values
    reps = int(np.ceil(horizon/7))
    preds = np.tile(last, reps)[:horizon]
    return preds

def moving_avg(df, window=7, horizon=30):
    last_mean = df.sort_values('date').tail(window)['units_sold'].mean()
    return np.repeat(last_mean, horizon)