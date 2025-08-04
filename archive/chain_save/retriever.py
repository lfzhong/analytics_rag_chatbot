from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from chain.preprocessor import load_raw_data, prepare_documents
from config.settings import EMBEDDING_MODEL_NAME, METRICS_FILE, INSIGHTS_FILE

import os

def get_raw_retriever(persist_dir="retriever_store", top_k=4):
    """
    Build a retriever from metric summary documents and return it along with sorted product and metric lists.
    """
    df_metrics, df_insights = load_raw_data(METRICS_FILE=METRICS_FILE, INSIGHTS_FILE=INSIGHTS_FILE)
    documents = prepare_documents(df_metrics, df_insights)

    filtered_docs = []
    products = set()
    metrics = set()

    for doc in documents:
        if doc.metadata.get("type") == "metric_summary":
            product = doc.metadata.get("product")
            metric = doc.metadata.get("metric")

            if product:
                products.add(product)
            if metric:
                metrics.add(metric)

            filtered_docs.append(Document(
                page_content=f"{metric} is a metric for product {product}.",
                metadata={"type": "definition", "product": product, "metric": metric}
            ))

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    vectorstore = Chroma.from_documents(
        documents=filtered_docs,
        embedding=embedding_model,
        persist_directory=persist_dir
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever, sorted(products), sorted(metrics)



def get_retriever_chain():
    raw_retriever = get_raw_retriever()
    return RunnableLambda(lambda x: x["question"]) | raw_retriever