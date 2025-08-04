from typing import Optional
import pandas as pd
import numpy as np
import os
from scipy.stats import linregress
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from config.settings import EMBEDDING_MODEL_NAME, METRICS_FILE, INSIGHTS_FILE, DATA_DIR
from langchain_huggingface import HuggingFaceEmbeddings
import logging

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TREND_SLOPE_TOLERANCE = 0.05

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def load_raw_data(METRICS_FILE=METRICS_FILE, INSIGHTS_FILE=INSIGHTS_FILE):
    """
    Load raw metrics and insights data from Excel files.
    """
    df_metrics = pd.read_excel(METRICS_FILE, sheet_name="Metrics")
    df_insights = pd.read_excel(INSIGHTS_FILE, sheet_name="Insights")
    return df_metrics, df_insights

def calculate_percentage_change(current_value, previous_value):
    """
    Calculate the percentage change between current and previous values.
    Returns np.nan if either value is NaN.
    """
    if pd.isna(current_value) or pd.isna(previous_value):
        return np.nan
    if previous_value == 0:
        if current_value > 0: return np.inf
        if current_value < 0: return -np.inf
        return 0.0
    return ((current_value - previous_value) / previous_value) * 100

def get_trend_description_lin_regress(values, tolerance=TREND_SLOPE_TOLERANCE):
    """
    Describe the trend of a series using linear regression slope.
    """
    clean_values = values.dropna()
    if len(clean_values) < 2:
        return "N/A (not enough data for trend)"
    x = np.arange(len(clean_values))
    y = clean_values.values
    slope, _, _, _, _ = linregress(x, y)
    if slope > tolerance:
        return "a clear increasing trend"
    elif slope < -tolerance:
        return "a clear decreasing trend"
    else:
        return "a relatively stable trend"

def prepare_documents(df_metrics: pd.DataFrame, df_insights: pd.DataFrame) -> list[Document]:
    """
    Prepare LangChain Document objects from metrics and insights DataFrames.
    """
    documents = []

    df_metrics_processed = df_metrics.melt(id_vars=["Product", "Metric"], var_name="Time", value_name="Value")
    df_metrics_processed["Time_parsed"] = pd.to_datetime(df_metrics_processed["Time"].astype(str), format="%Y-%m", errors="coerce")
    df_metrics_processed['Value'] = pd.to_numeric(df_metrics_processed['Value'], errors='coerce')
    df_metrics_processed.dropna(subset=["Value", "Time_parsed"], inplace=True)
    df_metrics_processed = df_metrics_processed.sort_values(by=["Product", "Metric", "Time_parsed"])

    for (product, metric), group_df in df_metrics_processed.groupby(["Product", "Metric"]):
        product_metric_str = f"{product} {metric}"
        product_metric_embedding = embedding_model.embed_query(product_metric_str)
        min_date = group_df['Time_parsed'].min()
        max_date = group_df['Time_parsed'].max()
        full_date_range = pd.date_range(start=min_date.to_period('M').start_time, end=max_date.to_period('M').end_time, freq='MS')
        series = group_df.set_index('Time_parsed')['Value'].reindex(full_date_range)

        for current_month_start_dt, current_value in series.items():
            if pd.isna(current_value):
                continue
            current_month_str = current_month_start_dt.strftime('%Y-%m')
            current_year = current_month_start_dt.year

            previous_month_value = series.get(current_month_start_dt - pd.DateOffset(months=1))
            mom_change_pct = calculate_percentage_change(current_value, previous_month_value)
            mom_desc = f" ({mom_change_pct:.2f}% MoM)" if pd.notna(mom_change_pct) and abs(mom_change_pct) != np.inf else ""

            previous_year_value = series.get(current_month_start_dt - pd.DateOffset(years=1))
            yoy_change_pct = calculate_percentage_change(current_value, previous_year_value)
            yoy_desc = f" ({yoy_change_pct:.2f}% YoY)" if pd.notna(yoy_change_pct) and abs(yoy_change_pct) != np.inf else ""

            six_month_values = series.loc[current_month_start_dt - pd.DateOffset(months=5): current_month_start_dt]
            ytd_values = series.loc[pd.Timestamp(f"{current_year}-01-01"): current_month_start_dt]

            doc_id = str(uuid.uuid4())
            metric_doc = Document(
                page_content=(
                    f"**Monthly Summary for {product} - {metric} in {current_month_str}:**\n"
                    f"Value: {current_value:.2f}{mom_desc}{yoy_desc}\n"
                    f"Past 6-Month Trend: {get_trend_description_lin_regress(six_month_values)}\n"
                    f"Year-to-Date (YTD) Trend ({current_year}): {get_trend_description_lin_regress(ytd_values)}"
                ),
                metadata={
                    "doc_id": doc_id,
                    "product": product,
                    "metric": metric,
                    "time_point": current_month_start_dt.isoformat(),
                    "month_year_str": current_month_str,
                    "month_year_int": int(current_month_str.replace("-", "")),  # Add this line for numeric filtering
                    "time_granularity": "monthly_summary",
                    "value": current_value,
                    "mom_change_pct": mom_change_pct,
                    "yoy_change_pct": yoy_change_pct,
                    "six_month_trend_desc": get_trend_description_lin_regress(six_month_values),
                    "ytd_trend_desc": get_trend_description_lin_regress(ytd_values),
                    "type": "metric_summary"
                }
            )
            documents.append(metric_doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if not df_insights.empty and "Month" in df_insights and "Insight Summary" in df_insights:
        df_insights["Month"] = pd.to_datetime(df_insights["Month"])
        for _, row in df_insights.iterrows():
            insight_text = f"Insight for {row['Month'].strftime('%Y-%m')}: {row['Insight Summary']}"
            chunks = splitter.create_documents([insight_text])
            for i, doc in enumerate(chunks):
                doc.metadata.update({
                    "doc_id": str(uuid.uuid4()),
                    "original_insight_id": str(uuid.uuid4()),
                    "month_year_str": row['Month'].strftime('%Y-%m'),
                    "month_year_int": int(row['Month'].strftime('%Y%m')),  # Add this line for numeric filtering
                    "month_datetime": row['Month'].isoformat(),
                    "type": "insight",
                    "chunk_idx": i
                })
                documents.append(doc)
    else:
        logging.warning("Insights DataFrame is empty or missing 'Month'/'Insight Summary' columns. No insight documents prepared.")

    schema_content = (
        f"This knowledge base contains time series data for various products and metrics, as well as past qualitative insights. "
        f"The primary metrics data (df_metrics) includes columns like: {', '.join(df_metrics.columns)}. "
        f"The past insights data (df_insights) includes columns like: {', '.join(df_insights.columns)}. "
        f"For each product and metric, monthly summaries are provided. These summaries include the current month's value, Month-over-Month (MoM %) change, Year-over-Year (YoY %) change, "
        f"a descriptive trend over the past 6 months (calculated using linear regression), and a Year-to-Date (YTD) trend for the current year (also based on linear regression). "
        f"This comprehensive data allows for analysis of product and metric performance, growth trajectories, and contextual historical events."
    )
    schema_doc = Document(
        page_content=schema_content,
        metadata={
            "doc_id": str(uuid.uuid4()),
            "type": "schema_overview",
            "source": "data_schema_description"
        }
    )
    documents.append(schema_doc)

    return documents