import os
import pandas as pd
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_core.documents import Document

from config.settings import (
    EMBEDDING_MODEL_NAME,
    QDRANT_PATH,
    QDRANT_COLLECTION_NAME,
    METRICS_FILE,
    INSIGHTS_FILE,
    VECTOR_STORE_TYPE  # Still used for logging consistency
)
from chain.preprocessor import prepare_documents

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def init_vector_store(documents: list[Document]):
    """
    Initializes and populates a local Qdrant vector store.
    """
    print(f"Initializing {VECTOR_STORE_TYPE} vector store...")
    try:
        client = QdrantClient(path=QDRANT_PATH)  # Local persistence
        vectorstore = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            client=client,
            collection_name=QDRANT_COLLECTION_NAME,
            force_recreate=True
        )
        print("✅ Qdrant vector store initialized.")
        return vectorstore
    except Exception as e:
        print(f"❌ Error initializing vector store: {e}")
        return None

def load_vector_store():
    """
    Loads an existing Qdrant vector store from disk if available.
    """
    print(f"Attempting to load {VECTOR_STORE_TYPE} vector store...")
    try:
        if not os.path.exists(QDRANT_PATH) or not os.listdir(QDRANT_PATH):
            print("❌ Qdrant data directory not found or empty.")
            return None

        client = QdrantClient(path=QDRANT_PATH)
        vectorstore = QdrantVectorStore.from_existing_collection(
            client=client,
            collection_name=QDRANT_COLLECTION_NAME,
            embedding=embeddings
        )
        print("✅ Qdrant vector store loaded.")
        return vectorstore
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        return None


def save_vector_store(vectorstore):
    """
    Qdrant handles local persistence automatically.
    """
    print("Saving Qdrant vector store...")
    try:
        if not os.path.exists(QDRANT_PATH):
            os.makedirs(QDRANT_PATH)
        print("✅ Qdrant vector store data persists automatically.")
    except Exception as e:
        print(f"❌ Error ensuring Qdrant data directory: {e}")

def load_or_create_vector_store():
    """
    Loads the existing vector store if available, otherwise rebuilds it from data.
    """
    print(f"🔍 Checking for existing {VECTOR_STORE_TYPE} vector store...")
    vectorstore = load_vector_store()
    if vectorstore is not None:
        print("✅ Vector store loaded successfully.")
        return vectorstore

    print("⚠️ No existing vector store found. Rebuilding from data...")
    try:
        df_metrics = pd.read_excel(METRICS_FILE, sheet_name="Metrics")
        df_insights = pd.read_excel(INSIGHTS_FILE, sheet_name="Insights")
        documents = prepare_documents(df_metrics, df_insights)
        vectorstore = init_vector_store(documents)
        save_vector_store(vectorstore)
        print("✅ New vector store created and saved.")
        return vectorstore
    except Exception as e:
        print(f"❌ Error during vector store rebuild: {e}")
        return None
