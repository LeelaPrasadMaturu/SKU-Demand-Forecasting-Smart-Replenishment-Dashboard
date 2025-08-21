# app/streamlit_app.py - Streamlit dashboard for SKU forecasting & replenishment (run locally)

import streamlit as st
import pandas as pd
import numpy as np
from utils import load_data, trailing_stats, compute_reorder_point, moving_avg, seasonal_naive, load_prophet_model, forecast_with_prophet, load_xgb_model, forecast_with_xgb
import joblib
import io

st.set_page_config(page_title="SKU Demand Forecast & Smart Replenishment", layout="wide")

# Dataset selector (future extensibility)
dataset_option = st.sidebar.selectbox("Select dataset", ["Synthetic (default)"])
sales, meta = load_data(sales_path='data/synthetic_sales.csv', meta_path='data/sku_metadata.csv')
sales['date'] = pd.to_datetime(sales['date'])


st.sidebar.header("Controls")
all_skus = sorted(sales['sku_id'].unique().tolist())
selected_skus = st.sidebar.multiselect("Select SKUs", all_skus, default=[all_skus[0]])
horizon = st.sidebar.number_input("Forecast horizon (days)", min_value=7, max_value=90, value=30)
service_level = st.sidebar.selectbox("Service level", ["90%","95%","99%"])
z_map = {"90%":1.282, "95%":1.645, "99%":2.33}
z = z_map[service_level]

# Forecast method selector
forecast_method = st.sidebar.selectbox("Forecast method", ["Statistical Baseline", "Prophet Model", "XGBoost Model"])

lead_time_override = {}
for sku in selected_skus:
    default_lt = int(meta.loc[meta['sku_id']==sku, 'lead_time_days'].values[0])
    lead_time_override[sku] = st.sidebar.number_input(f"Lead time for {sku}", min_value=1, max_value=60, value=default_lt, key=f"lt_{sku}")


st.title("SKU Demand Forecast & Smart Replenishment")


rows = []
for sku in selected_skus:
    df_sku = sales[sales['sku_id']==sku].sort_values('date').copy()
    as_of = df_sku['date'].max()
    avg_daily, std_daily = trailing_stats(sales, sku, as_of, window_days=30)
    lead_time = lead_time_override[sku]
    reorder_point = compute_reorder_point(avg_daily, lead_time, std_daily, z=z)
    current_stock = int(meta.loc[meta['sku_id']==sku, 'init_stock'].values[0])
    days_until_stockout = (current_stock / avg_daily) if avg_daily > 0 else np.inf
    alert = (current_stock <= reorder_point) or (days_until_stockout <= lead_time)
    # EOQ calculation
    ordering_cost = float(meta.loc[meta['sku_id']==sku, 'ordering_cost'].values[0])
    holding_cost = float(meta.loc[meta['sku_id']==sku, 'holding_cost_per_unit_per_year'].values[0])
    unit_cost = float(meta.loc[meta['sku_id']==sku, 'unit_cost'].values[0])
    D = avg_daily * 365
    S = ordering_cost
    H = holding_cost if holding_cost > 0 else 0.1 * unit_cost
    eoq = np.sqrt((2 * D * S) / H) if H > 0 else 0
    rows.append({
        'sku': sku,
        'current_stock': current_stock,
        'avg_daily': round(avg_daily,2),
        'std_daily': round(std_daily,2),
        'lead_time': lead_time,
        'reorder_point': int(np.ceil(reorder_point)),
        'days_until_stockout': round(days_until_stockout,2) if np.isfinite(days_until_stockout) else 'inf',
        'alert': 'YES' if alert else 'NO',
        'EOQ': int(np.ceil(eoq))
    })

kpi_df = pd.DataFrame(rows)

# KPI tiles
st.markdown("### KPI Overview")
cols = st.columns(min(5, len(selected_skus)))
for i, sku in enumerate(selected_skus):
    kpi = kpi_df[kpi_df['sku']==sku].iloc[0]
    with cols[i % len(cols)]:
        st.metric(label=f"SKU: {sku}", value=kpi['current_stock'], help="Current stock")
        st.metric(label="Reorder Point", value=kpi['reorder_point'])
        st.metric(label="Days Until Stockout", value=kpi['days_until_stockout'])
        st.metric(label="EOQ", value=kpi['EOQ'])

st.subheader("Replenishment KPIs Table")
st.dataframe(kpi_df)

to_reorder = kpi_df[kpi_df['alert']=='YES'].sort_values('days_until_stockout')
st.subheader("Top SKUs to reorder (alerts)")
st.dataframe(to_reorder)


if len(selected_skus) > 0:
    st.subheader(f"Forecast for selected SKU")
    sku0 = selected_skus[0]
    df_sku0 = sales[sales['sku_id']==sku0].sort_values('date')
    # Prophet forecast
    prophet_model = load_prophet_model(sku0)
    prophet_vals = None
    if prophet_model is not None:
        prophet_df = forecast_with_prophet(prophet_model, df_sku0[['date','units_sold']].rename(columns={'date':'ds','units_sold':'y'}), periods=horizon)
        prophet_vals = prophet_df['yhat'].values
    # XGBoost forecast
    xgb_model = load_xgb_model(sku0)
    xgb_vals = None
    if xgb_model is not None:
        # Feature engineering for XGBoost (same as in training)
        df_xgb = df_sku0.copy()
        df_xgb['lag_1'] = df_xgb['units_sold'].shift(1)
        df_xgb['lag_7'] = df_xgb['units_sold'].shift(7)
        df_xgb['lag_14'] = df_xgb['units_sold'].shift(14)
        df_xgb['lag_28'] = df_xgb['units_sold'].shift(28)
        df_xgb['rolling_mean_7'] = df_xgb['units_sold'].rolling(7).mean().shift(1)
        df_xgb['rolling_std_14'] = df_xgb['units_sold'].rolling(14).std().shift(1)
        df_xgb['day_of_week'] = df_xgb['date'].dt.weekday
        df_xgb['is_weekend'] = df_xgb['day_of_week'].isin([5,6]).astype(int)
        df_xgb = df_xgb.dropna().reset_index(drop=True)
        features = ['lag_1','lag_7','lag_14','lag_28','rolling_mean_7','rolling_std_14','day_of_week','is_weekend','on_promo','price','views']
        xgb_vals = forecast_with_xgb(xgb_model, df_xgb, features, periods=horizon)
    # Prepare DataFrame for plotting
    actual = df_sku0.set_index('date')['units_sold']
    last_actual_date = df_sku0['date'].max()
    future_idx = pd.date_range(last_actual_date + pd.Timedelta(days=1), periods=horizon, freq='D')
    plot_df = pd.DataFrame({"actual": actual})
    if prophet_vals is not None:
        plot_df["Prophet forecast"] = pd.Series(prophet_vals, index=future_idx)
    if xgb_vals is not None:
        plot_df["XGBoost forecast"] = pd.Series(xgb_vals, index=future_idx)
    st.line_chart(plot_df)
    if prophet_vals is None:
        st.warning(f"No Prophet model found for {sku0}.")
    if xgb_vals is None:
        st.warning(f"No XGBoost model found for {sku0}.")
    # Explanation text
    st.markdown("""
    **Formulae used:**
    - Safety Stock = z × std_daily × sqrt(lead_time)
    - Reorder Point = avg_daily × lead_time + safety_stock
    - EOQ = sqrt((2 × annual_demand × ordering_cost) / holding_cost)
    
    _Example:_
    If avg_daily = 12.4, std_daily = 5.2, lead_time = 7, z = 1.645:
    - safety_stock = 1.645 × 5.2 × sqrt(7) ≈ 22.6
    - reorder_point = 12.4 × 7 + 22.6 ≈ 109.4
    """)


def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

def to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='ReorderList')
    return output.getvalue()

col1, col2 = st.columns(2)
with col1:
    if st.button("Export reorder CSV"):
        csv_bytes = to_csv_bytes(to_reorder)
        st.download_button(label="Download reorder list", data=csv_bytes, file_name='reorder_list.csv', mime='text/csv')
with col2:
    if st.button("Export weekly report (Excel)"):
        excel_bytes = to_excel_bytes(to_reorder)
        st.download_button(label="Download weekly report", data=excel_bytes, file_name='weekly_report.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')