import streamlit as st
from utils import extract_products_and_metrics, format_bullet_points
from config.settings import EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rag_pipeline import RAGPipeline
from archive.feedback_logger import FeedbackLogger
import uuid
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_PROJECT"] = "analytics-rag-chatbot_0706"


# Set up session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []
feedback_logger = FeedbackLogger(db_path="archive/feedback_log.db")
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

st.title("Marketing Analytics Chatbot")

# Initialize models and RAG pipeline
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory=str(PERSIST_DIRECTORY)
)
available_products, available_metrics = extract_products_and_metrics(vectorstore)
print(f"Available products: {len(available_products)}, Available metrics: {len(available_metrics)}")

# Initialize RAG pipeline with optional score threshold
# Using distance metrics where LOWER scores = MORE similar (e.g., 0.5 is better than 2.0)
# Good distance thresholds: 0.5 (very strict), 1.0 (moderate), 1.5 (lenient), 2.0 (very lenient)
score_threshold = 1.0  # Only include documents with distance <= 1.0
rag_pipeline = RAGPipeline(vectorstore, available_products, available_metrics, score_threshold)

# Chat UI: display chat history and handle new user input
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("Ask me anything about your analytics data..."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        response = rag_pipeline.get_rag_response(user_input)
        display_answer = format_bullet_points(response)
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").markdown(display_answer)
    # Feedback UI
    feedback = st.text_input("Leave feedback about this answer (optional):", key=f"feedback_{len(st.session_state['messages'])}")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("👍", key=f"thumbs_up_{len(st.session_state['messages'])}"):
            feedback_logger.log(
                question=user_input,
                answer=response,
                context="",  # Optionally pass context if available
                feedback=feedback,
                llm_model="llama3",
                prompt_template_name="reasoning_prompt.txt",
                session_id=st.session_state["session_id"],
                rating="positive",
                retrieved_doc_ids=""
            )
            st.success("Thank you for your positive feedback!")
    with col2:
        if st.button("👎", key=f"thumbs_down_{len(st.session_state['messages'])}"):
            feedback_logger.log(
                question=user_input,
                answer=response,
                context="",  # Optionally pass context if available
                feedback=feedback,
                llm_model="llama3",
                prompt_template_name="reasoning_prompt.txt",
                session_id=st.session_state["session_id"],
                rating="negative",
                retrieved_doc_ids=""
            )
            st.warning("Thank you for your feedback! We'll use it to improve.")

