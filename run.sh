#!/bin/bash
python notebooks/1_data_generation.ipynb
python notebooks/2_eda_and_preprocessing.ipynb
python notebooks/3_modeling_forecast.ipynb
streamlit run app/streamlit_app.py
