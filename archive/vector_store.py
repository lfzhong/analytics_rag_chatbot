from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd


def prepare_documents(df_metrics, df_insights):
    df = df_metrics.melt(id_vars=["Product", "Metric"], var_name="Time", value_name="Value")
    df["Time_parsed"] = pd.to_datetime(df["Time"].astype(str),  format="%Y-%m", errors="coerce")
    df.dropna(subset=["Value"], inplace=True)

    metric_docs = [
        Document(
            page_content=f"{row['Product']} - {row['Metric']} - {row['Time_parsed'].strftime('%Y-%m')}: {row['Value']}",
            metadata={"doc_id": f"metric_{i}"}
        )
        for i, (_, row) in enumerate(df.iterrows()) if pd.notna(row['Time_parsed'])
    ]

    insight_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    if not df_insights.empty and "Month" in df_insights and "Insight Summary" in df_insights:
        df_insights["Month"] = pd.to_datetime(df_insights["Month"])
        insight_texts = [
            f"Month: {row['Month'].strftime('%Y-%m')}, Insight: {row['Insight Summary']}"
            for _, row in df_insights.iterrows()
        ]
        insight_docs = splitter.create_documents(insight_texts)
        for i, doc in enumerate(insight_docs):
            doc.metadata["doc_id"] = f"insight_{i}"

    return metric_docs + insight_docs

def init_vector_store(documents):
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(documents, embedder)
