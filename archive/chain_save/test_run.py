from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_ollama import ChatOllama
from config.settings import *
import pandas as pd
from preprocessor import prepare_documents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# -----------------------------
# Load data and initialize vector store
# -----------------------------
df_metrics = pd.read_excel(METRICS_FILE, sheet_name="Metrics")
df_insights = pd.read_excel(INSIGHTS_FILE, sheet_name="Insights")
documents = prepare_documents(df_metrics, df_insights)

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model
)

retriever = vectorstore.as_retriever()

# -----------------------------
# Define LLM
# -----------------------------
llm = ChatOllama(model="llama3", temperature=0, streaming=True)

# -----------------------------
# Intent Chain
# -----------------------------
intent_prompt = PromptTemplate.from_template(intent_prompt_path.read_text())
intent_parser = JsonOutputParser()
intent_chain = intent_prompt | llm | intent_parser

# -----------------------------
# Reasoning Chain
# -----------------------------
from langchain_core.prompts import ChatPromptTemplate

reasoning_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful data analyst assistant. Use the data below to answer questions about metrics and insights."),
    ("human", "User Question: {question}\n\nRelevant Data:\n{retrieved_docs}\n\nYour Answer:")
])

reasoning_parser = StrOutputParser()
reasoning_chain = reasoning_prompt | llm | reasoning_parser

# -----------------------------
# Full Pipeline (No Metric Mapping)
# -----------------------------
def run_pipeline(question: str):
    print(f"\n🧠 Question: {question}")

    # Step 1: Extract Intent
    intent = intent_chain.invoke({"question": question})
    print("✅ Extracted intent:", intent)

    # Step 2: Build query from intent
    keywords = intent.get("metrics", []) + intent.get("products", []) + [intent.get("time_range", "")]
    query = " ".join(keywords)

    # Step 3: Retrieve relevant docs
    retrieved_docs = retriever.invoke(query)
    doc_text = "\n".join([doc.page_content for doc in retrieved_docs])
    print("doc_text:", doc_text)

    # Step 4: Generate answer
    answer = reasoning_chain.invoke({
        "question": question,
        "retrieved_docs": doc_text
    })

    print("📊 Final Answer:\n", answer)
    return answer

# -----------------------------
# Run the pipeline
# -----------------------------
question = "how does TH visits look like in May 2025?"
run_pipeline(question)
