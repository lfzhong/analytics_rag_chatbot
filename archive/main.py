# main.py

import pandas as pd
import os
from chain.preprocessor import prepare_documents
from archive.src.vector_store_utils import init_vector_store, save_vector_store, load_vector_store, FAISS_INDEX_PATH
from archive.src.llm_interface import create_overall_rag_chain  # Import the chain builder function
from config.settings import METRICS_FILE, INSIGHTS_FILE, DATA_DIR
# Note: extract_filters_from_query_llm and query_rag_system are NOT imported directly
# because the overall_rag_chain will handle all the steps internally.
# query_rag_system is just a helper wrapper for invoking the chain.


# # --- Configuration ---
# # Define paths for your data files relative to the main.py script
# DATA_DIR = "data"
# METRICS_FILE = os.path.join(DATA_DIR, "metrics_data.xlsx")
# INSIGHTS_FILE = os.path.join(DATA_DIR, "insights_data.xlsx")


def run_rag_application():
    """
    Orchestrates the entire RAG application workflow:
    1. Loads data.
    2. Prepares documents.
    3. Manages (creates or loads) the FAISS vector store.
    4. Sets up the LangChain RAG pipeline.
    5. Enters an interactive loop for user queries.
    """

    # 1. Load Data
    print("Loading data...")
    df_metrics = pd.DataFrame()  # Initialize empty DataFrames
    df_insights = pd.DataFrame()

    try:
        if not os.path.exists(DATA_DIR):
            print(f"Error: Data directory '{DATA_DIR}' not found.")
            print("Please create it and place 'metrics_data.xlsx' and 'insights_data.xlsx' inside.")
            return

        if not os.path.exists(METRICS_FILE):
            print(f"Warning: Metrics file '{METRICS_FILE}' not found. Continuing without metrics data.")
        else:
            df_metrics = pd.read_excel(METRICS_FILE, sheet_name="Metrics")

        if not os.path.exists(INSIGHTS_FILE):
            print(f"Warning: Insights file '{INSIGHTS_FILE}' not found. Continuing without insights data.")
        else:
            df_insights = pd.read_excel(INSIGHTS_FILE, sheet_name="Insights")

        if df_metrics.empty and df_insights.empty:
            print("No data loaded. Please ensure data files exist and are correctly formatted.")
            return

        print("Data loading check complete. Proceeding with available data.")

    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return

    # 2. Prepare Documents
    print("Preparing documents...")
    documents = prepare_documents(df_metrics, df_insights)
    print(f"Prepared {len(documents)} documents.")

    if not documents:
        print("No documents were prepared. Cannot proceed with vector store creation.")
        return

    # 3. Create or Load Vector Store
    vectorstore = None
    if os.path.exists(FAISS_INDEX_PATH):
        vectorstore = load_vector_store()

    if vectorstore is None:  # If not loaded or error, create new
        print("Vector store not found or failed to load. Creating a new one...")
        vectorstore = init_vector_store(documents)
        save_vector_store(vectorstore)  # Save it for future runs

    if vectorstore is None:  # Critical error if still none after attempt
        print("Failed to initialize or load vector store. Exiting application.")
        return

    # 4. Set Up LangChain RAG Pipeline
    print("\nSetting up RAG pipeline...")
    # Pass the loaded vectorstore to the chain builder
    full_rag_chain = create_overall_rag_chain(vectorstore)
    print("RAG pipeline setup complete.")

    # 5. Interactive Query Loop
    print("\n--- RAG System Ready for Queries ---")
    print("Type 'exit' to quit.")

    while True:
        user_question = input("\nEnter your question: ")
        if user_question.lower() == 'exit':
            print("Exiting RAG application. Goodbye!")
            break

        print("Processing your question...")
        try:
            # Invoke the full RAG chain with the user's question.
            # The chain itself handles intent, metadata extraction, retrieval, generation, validation, and formatting.
            response = full_rag_chain.invoke({"input": user_question, "question": user_question})

            # The final_answer is the key where the result from the last step of the chain is stored
            if "final_answer" in response:
                print(f"\nAnswer:\n{response['final_answer']}")
            else:
                # If final_answer is not present, it means an error occurred earlier and was propagated.
                # You might want to print more debug info here based on the 'response' dict.
                print("\nAn error occurred during processing. Here's the full response for debugging:")
                print(json.dumps(response, indent=2))

        except Exception as e:
            print(f"\nAn unexpected error occurred during query processing: {e}")
            print("Please try rephrasing your question or check the logs for details.")

        print("-" * 70)


if __name__ == "__main__":
    run_rag_application()