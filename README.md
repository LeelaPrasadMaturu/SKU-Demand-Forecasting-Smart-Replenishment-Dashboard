
# SKU Demand Forecasting & Smart Replenishment Dashboard

## Real-World Impact & Industry Adoption

**Companies using advanced inventory models:**
- **Nykaa** (beauty & personal care)
- Amazon
- Walmart
- Zara
- Unilever
- Target
- Flipkart
- Procter & Gamble

These companies leverage demand forecasting and smart replenishment to optimize stock, reduce costs, and improve customer satisfaction.

**Other Use Cases:**
- Grocery and supermarket chains (perishable inventory)
- Pharma and healthcare supply chains
- Electronics and mobile retail
- Automotive parts distribution
- Restaurant and QSR chains (ingredient management)
- E-commerce and omnichannel fulfillment

# Smart Inventory Replenishment System


## Project Overview
This project implements a robust, production-style SKU-level demand forecasting and smart inventory replenishment system for D2C brands and retailers. It generates synthetic sales data, fits multiple forecasting models (Prophet, XGBoost), computes inventory KPIs (EOQ, reorder point, safety stock), and provides a modern Streamlit dashboard for actionable insights and exportable reports.


---

## File & Folder Structure

```
project_root/
├─ data/
│  ├─ synthetic_sales.csv         # Synthetic sales data (10 SKUs × 180 days, easily scalable)
│  └─ sku_metadata.csv            # SKU metadata (category, lead time, stock, cost, etc.)
├─ notebooks/
│  ├─ 1_data_generation.ipynb     # Generates and saves synthetic data
│  ├─ 2_eda_and_preprocessing.ipynb # EDA, cleaning, SKU classification
│  └─ 3_modeling_forecast.ipynb   # Baseline, Prophet, XGBoost, metrics, model saving
├─ app/
│  ├─ streamlit_app.py            # Streamlit dashboard (forecast, alerts, export)
│  └─ utils.py                    # Data loading, inventory formulas, helpers
├─ models/
│  ├─ xgb_SKU_xx.joblib           # XGBoost model artifact (per SKU)
│  └─ prophet_models/             # Prophet model artifacts (per SKU)
├─ requirements.txt               # All dependencies
├─ run.sh                         # Convenience script to run notebooks/app
├─ README.md                      # This file
├─ demo_script.txt                # 60–90s demo narration
├─ resume_bullets.txt             # Resume bullet variants
├─ checklist.txt                  # MVP and future goals checklist
└─ test_smoke.py                  # Minimal smoke test
```


---

## Key Features
- **SKU-level demand forecasting** using Prophet and XGBoost
- **EOQ, reorder point, and safety stock** calculations
- **Streamlit dashboard** for interactive exploration, alerts, and export
- **Exportable reports** (CSV, Excel) for supply chain teams
- **Modular codebase** for easy extension to new models or data sources

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Train models and generate data: run the notebooks in `notebooks/`
3. Launch the dashboard: `streamlit run app/streamlit_app.py`

---

## Why This Matters
Inventory optimization is a core driver of profitability and customer experience for modern retailers and D2C brands. Companies like **Nykaa** and Amazon use similar techniques to:
- Minimize stockouts and overstock
- Automate replenishment
- Respond to demand spikes and seasonality
- Improve working capital efficiency

This project demonstrates these best practices in a transparent, hands-on way.

## How to Run

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Generate Data & Run Notebooks
Run each notebook in `notebooks/` sequentially, or use:
```bash
python notebooks/1_data_generation.ipynb
python notebooks/2_eda_and_preprocessing.ipynb
python notebooks/3_modeling_forecast.ipynb
```

### 3. Launch the Dashboard
```bash
streamlit run app/streamlit_app.py
```
Open the provided local URL in your browser.

### 4. Run Smoke Test
```bash
python test_smoke.py
```

---

## Expected Outputs
- **data/synthetic_sales.csv**: 10 SKUs × 180 days of realistic sales, price, promo, and views data (easily scalable to 100k+ rows).
- **data/sku_metadata.csv**: SKU-level info (category, lead time, stock, cost, etc.).
- **notebooks/**: EDA, plots, per-SKU stats, model metrics (MAPE, RMSE), and saved models.
- **app/streamlit_app.py**: Dashboard with:
	- Multi-SKU selection, forecast horizon, service level, lead time override
	- KPI tiles: current stock, reorder point, days until stockout, EOQ
	- Forecast plot (history + forecast)
	- Reorder alerts table (with EOQ)
	- Export reorder list (CSV/Excel)
	- Explanation of formulas
- **models/**: Saved Prophet and XGBoost models for reproducibility.
- **demo_script.txt**: 60–90s demo narration for video or live demo.
- **resume_bullets.txt**: Three resume bullet variants.
- **checklist.txt**: MVP and future goals.

---

## Checklist (MVP & Future Goals)

### MVP (Completed)
- [x] Synthetic data generation (10 SKUs × 180 days, scalable)
- [x] EDA, cleaning, SKU classification
- [x] Baseline, Prophet, and XGBoost models (metrics, model saving)
- [x] Inventory KPIs (EOQ, reorder point, safety stock)
- [x] Streamlit dashboard (forecast, alerts, export)
- [x] Resume bullets, demo script, smoke test

### Future Goals
- [ ] Scale data to 100k+ records for true enterprise simulation
- [ ] Add ARIMA and LSTM forecasting options
- [ ] Implement buffer stock and advanced business rules
- [ ] Auto-generate and email weekly PDF/Excel reports
- [ ] Deploy dashboard to Heroku/Render/Streamlit Cloud
- [ ] Add user authentication and multi-user support
- [ ] Integrate real supplier APIs for lead time updates
- [ ] Add unit tests and CI/CD pipeline
- [ ] Add Dockerfile for containerized deployment

---

## Assumptions & Notes
- All data is synthetic and reproducible (np.random.seed=42).
- All code is offline and does not require API keys.
- The project is modular and can be extended for real-world use.

