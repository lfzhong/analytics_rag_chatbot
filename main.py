from utils import analyze_question, extract_products_and_metrics, build_chroma_filter
from config.settings import EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from chain.intent_classification.intent_classifier import IntentClassifier
from chain.query_parsing.query_parser import QueryParser
from chain.answer_generation.answer_generator import AnswerGenerator
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Use the PERSIST_DIRECTORY from settings
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory=str(PERSIST_DIRECTORY)
)

# Initialize Ollama LLM (llama3)
llm = OllamaLLM(model="llama3")

# Initialize chains
intent_chain = IntentClassifier()
query_chain = QueryParser()
reasoning_chain = AnswerGenerator(model_name="llama3")

# Dynamically extract available products and metrics from the vectorstore
available_products, available_metrics = extract_products_and_metrics(vectorstore)
format_instructions = query_chain.format_instructions  # Get from the chain's parser

question_list = [
    "How does TH active 30% look like in Q1 2025?"
    # "How does TH utilization rate look like in Q1 2024 and Q1 2025?",
    # "what is the value of CCM active 30% for April 2025",
    # "What happened to active 30 for both TH and CCM in May 2025",
                # "How’s everything looking lately?",
                # "Can I get a quick health check on TH?",
                # "Are we seeing any performance issues with CCM?",
                # "What are the concerning metrics from Q1?",
                # "Anything unusual in the latest numbers?",
                # "That makes sense."
    ]

for q in question_list:
    print("Question:\n", q)
    # Step 1: Get intent result
    intent_result = intent_chain.run(q)
    print("Intent result:", intent_result)

    # Step 2: Pass intent result and question to query_chain
    query_result = query_chain.run({
        "question": q,
        "intent": intent_result,
        "available_products": available_products,
        "available_metrics": available_metrics,
        "format_instructions": format_instructions
    })
    print("Query result:", query_result)

    # Step 3: Use query_result to construct semantic search query
    products = getattr(query_result, 'products', []) or [""]
    metrics = getattr(query_result, 'metrics', []) or [""]
    time_spans = getattr(query_result, 'time_spans', [])
    query_concepts = [f"{p} {m}".strip() for p in products for m in metrics]
    query_str = query_concepts[0] if query_concepts else q
    print("Query string:", query_str)

    # Step 4: Semantic search by text, filter only by product and time (not metric)
    chroma_filter = build_chroma_filter(products, [], time_spans)  # No metric filter
    print("chroma_filter:", chroma_filter)
    results = vectorstore.similarity_search(query_str, filter=chroma_filter, k =10)

    # Step 5: Aggregate context and reasoning
    metric_docs = [doc for doc in results if doc.metadata.get("type") == "metric_summary"]
    insight_docs = [doc for doc in results if doc.metadata.get("type") == "insight"]

    context = ""
    if insight_docs:
        context += "PAST INSIGHTS:\n" + "\n".join(doc.page_content for doc in insight_docs) + "\n"
    if metric_docs:
        context += "METRIC SUMMARIES:\n" + "\n".join(doc.page_content for doc in metric_docs)
    # Print out metric summaries for debugging
    print("\n[DEBUG] Metric Summaries:")
    for doc in metric_docs:
        print(doc.page_content)
    answer = reasoning_chain.run(q, context)
    print("\nLLM Answer:\n", answer)

