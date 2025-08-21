# Minimal smoke test for Smart Inventory Replenishment System
import pandas as pd
import os

# Test data files
assert os.path.exists('data/synthetic_sales.csv'), 'synthetic_sales.csv missing'
assert os.path.exists('data/sku_metadata.csv'), 'sku_metadata.csv missing'
sales = pd.read_csv('data/synthetic_sales.csv')
meta = pd.read_csv('data/sku_metadata.csv')
assert not sales.isnull().any().any(), 'NaNs in sales data'
assert not meta.isnull().any().any(), 'NaNs in metadata'
print('Data files present and valid.')

# Test model artifacts
models_dir = 'models'
assert os.path.exists(models_dir), 'models/ directory missing'
print('Models directory present.')

# Test Streamlit app file
assert os.path.exists('app/streamlit_app.py'), 'streamlit_app.py missing'
print('Streamlit app present.')

print('Smoke test passed.')
