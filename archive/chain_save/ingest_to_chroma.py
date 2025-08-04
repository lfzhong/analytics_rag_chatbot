# ingest_to_chroma.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chain.preprocessor import load_raw_data, prepare_documents
from config.settings import EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def main():
    # 1. Load your raw data
    df_metrics, df_insights = load_raw_data()

    # 2. Prepare LangChain Document objects
    documents = prepare_documents(df_metrics, df_insights)

    # 3. Create embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 4. Create Chroma vectorstore and add documents
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(PERSIST_DIRECTORY)
    )
    print(f"Vectorstore created with {len(documents)} documents at {PERSIST_DIRECTORY}")

if __name__ == "__main__":
    main()