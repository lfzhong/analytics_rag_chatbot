from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chain.preprocessor import load_raw_data, prepare_documents
from config.settings import EMBEDDING_MODEL_NAME

# Load and prepare documents
df_metrics, df_insights = load_raw_data()
documents = prepare_documents(df_metrics, df_insights)

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Initialize Chroma
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="./chroma_store"  # Optional for saving
)

# Persist the vectorstore to disk after creation
# vectorstore.persist()
