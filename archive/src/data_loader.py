# src/data_loader.py
import pandas as pd
from config.settings import METRICS_FILE, INSIGHTS_FILE

def load_metrics_data(file_path=METRICS_FILE):
    df = pd.read_excel(file_path, sheet_name='Metrics Data')
    return df

def load_insights_data(file_path=INSIGHTS_FILE):
    df = pd.read_excel(file_path, sheet_name='Insights')
    return df