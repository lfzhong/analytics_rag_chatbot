# src/llm_interface.py

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_core.documents import Document
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import json
import re
from difflib import get_close_matches
from typing import List, Dict, Any, Optional

# --- Global LLM instance (ensure llama3 is pulled in Ollama) ---
llm = OllamaLLM(model="llama3")

# --- 1. Intent classification ---
intent_prompt = PromptTemplate.from_template("""
Classify the following user input as either "question" or "feedback". Respond with ONLY one of these words, and nothing else.

If the input contains a question about data or analysis, classify as "question".
If the input is a reaction, suggestion, or comment about the assistant's output, classify as "feedback".

Input: {input}
Output:
""")
intent_chain = intent_prompt | llm

# --- 2. Structured Metadata Extraction ---
understand_prompt = PromptTemplate.from_template("""
You are an intelligent assistant helping analyze business questions.

Your job is to extract the following from the user's question:
- metrics: The key metric(s) mentioned (as a list of strings), or null if none. Match exactly to known metrics if possible (e.g., "Sales", "Profit", "Customer Acquisition Cost").
- products: The product(s) mentioned (as a list of strings), or null if none. Match exactly to known products (e.g., "Product A", "Product B").
- time_point_prefix: A string representing the year and month if a specific month is mentioned (e.g., "2024-03" for March 2024), or "YYYY" for a full year (e.g., "2023"). Use null if no specific time point is provided, or if the range is general like "last quarter".
- analysis_type: One of ["summary", "trend", "anomaly", "comparison", "insight"], or null. "summary" for specific values, "trend" for changes over time, "anomaly" for unusual data points, "comparison" for comparing entities, "insight" if the user seems to be asking for general qualitative observations.

Return your output as **strictly valid JSON** with exactly these keys. Do not include any explanation, markdown, or formatting.

User Question:
{question}

Output:
""")

understand_chain = understand_prompt | llm

# --- 3. Clean and Validate Extracted Metadata ---
def clean_and_validate_metadata(llm_output: str) -> dict:
    """
    Cleans and validates the JSON output from the understanding LLM.
    Converts time_point to the format expected by your document metadata if possible.
    """
    try:
        data = json.loads(llm_output)
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON from understand_chain: {llm_output}")
        return {"error": "Invalid JSON format from LLM for metadata extraction."}

    cleaned = {}
    for k, v in data.items():
        key = k.strip().lower()
        if isinstance(v, str):
            cleaned[key] = v.strip()
        elif isinstance(v, list):
            cleaned[key] = [item.strip() for item in v]
        else:
            cleaned[key] = v

    if 'analysis_type' in cleaned:
        if cleaned['analysis_type'] == 'summary':
            cleaned['type'] = 'metric_summary'
        elif cleaned['analysis_type'] == 'insight':
            cleaned['type'] = 'insight'
        del cleaned['analysis_type']

    for key in ["metrics", "products"]:
        if key in cleaned and (cleaned[key] is None or len(cleaned[key]) == 0):
            cleaned[key] = None

    return cleaned

clean_metadata_chain = RunnableLambda(clean_and_validate_metadata)

# --- 4. Define the Retrieval Chain (incorporates metadata filtering) ---
# This setup requires the vectorstore to be passed in.

def setup_rag_chain(vectorstore):
    """
    Sets up the core RAG chain for retrieval and generation.
    This chain will receive the extracted filters and query from the user.
    """
    rag_prompt = PromptTemplate.from_template("""
    You are an experienced AI analytics assistant.

    Using the provided context, generate 3–5 concise business insights.
    Prioritize providing exact numerical values and specific timeframes when available.

    Guidelines:
    - Mimic the style and tone of the provided context if it contains insights.
    - NEVER include metrics, products, or timeframes not explicitly supported by the context.
    - Focus on recent trends when comparing months; explicitly name the months and values.
    - Reference actual metric values, product names, and timeframes.
    - If the context is too sparse or inconclusive for the user's question, state this instead of guessing.

    User Question:
    {question}

    Relevant Data Context:
    {context}

    Insights:
    """)

    llm_rag = OllamaLLM(model="llama3")
    document_chain = create_stuff_documents_chain(llm_rag, rag_prompt)
    return document_chain

def create_retriever_with_filters(vectorstore, meta: Dict[str, Any], relax_filters: bool = False):
    """
    Creates a retriever from the vector store with optional relaxed metadata filters.
    """
    search_kwargs = {"k": 10}
    # Metadata filters
    # filters = {}
    #
    # if meta.get("type"):
    #     filters["type"] = meta["type"]
    #
    # if not relax_filters:
    #     if meta.get("products"):
    #         if isinstance(meta["products"], list) and meta["products"]:
    #             filters["product"] = meta["products"][0]
    #     if meta.get("metrics"):
    #         if isinstance(meta["metrics"], list) and meta["metrics"]:
    #             filters["metric"] = meta["metrics"][0]
    #     if meta.get("time_point_prefix"):
    #         filters["month_year_str"] = meta["time_point_prefix"]
    #
    # if filters:
    #     search_kwargs["filter"] = filters
    #     print(f"Retriever search_kwargs with filter: {search_kwargs}")
    # else:
    #     print("No specific metadata filters extracted.")

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    return retriever


# --- 5. Custom Retrieval and Error Handling Chain ---
def retrieve_and_validate(inputs: Dict[str, Any], vectorstore) -> Dict[str, Any]:
    """
    Custom function to retrieve documents and handle potential errors or missing data.
    It passes through the original question and extracted metadata.
    """
    question = inputs["question"]
    meta = inputs["metadata"]

    if meta.get("error"):
        return {"error": meta["error"], "question": question, "retrieved_data": []}

    retriever = create_retriever_with_filters(vectorstore, meta, relax_filters=True)

    print(f"DEBUG: Retriever input (question): {question}") # DEBUG
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        print(f"DEBUG: No documents retrieved for question: {question} with filters: {meta}") # DEBUG
        return {
            "error": "No relevant documents found in the vector store based on your query and filters.",
            "question": question,
            "retrieved_data": [],
            "meta": meta
        }

    retrieved_data_str = "\n".join([f"Document Content: {doc.page_content}\nMetadata: {doc.metadata}" for doc in retrieved_docs])

    print(f"DEBUG: Retrieved Data String (for LLM context - first 200 chars):\n{retrieved_data_str[:200]}...") # DEBUG

    return {
        "retrieved_documents": retrieved_docs,
        "retrieved_data": retrieved_data_str, # String for LLM
        "meta": meta,
        "question": question
    }

# --- 6. Check for exact or fuzzy metric match (Now on retrieved Documents) ---
def check_metrics_and_suggest_on_docs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    metadata = inputs.get("meta", {})
    requested_metrics = metadata.get("metrics", [])
    retrieved_documents = inputs.get("retrieved_documents", [])

    available_metrics_in_docs = set()
    for doc in retrieved_documents:
        if "metric" in doc.metadata:
            available_metrics_in_docs.add(doc.metadata["metric"])

    available_lower = {m.lower() for m in available_metrics_in_docs if m}
    requested_lower = [m.lower() for m in requested_metrics if m]

    missing = [m for m in requested_lower if m not in available_lower]

    if not missing:
        return inputs

    suggestions = {}
    for m_lower in missing:
        match = get_close_matches(m_lower, list(available_lower), n=1, cutoff=0.7)
        if match:
            original_requested_metric = next((rm for rm in requested_metrics if rm.lower() == m_lower), m_lower)
            original_available_metric = next((am for am in available_metrics_in_docs if am.lower() == match[0]), match[0])
            suggestions[original_requested_metric] = original_available_metric

    error_message = (
        f"The following requested metrics were not clearly found in the retrieved data: {', '.join(missing)}. "
        f"Closest available metrics: {suggestions if suggestions else 'None'}. "
        "Therefore, insights might be incomplete or missing."
    )

    inputs["warning"] = error_message
    return inputs


# --- PROMPT DEFINITIONS (Moved up to avoid NameError) ---
# 7. Reasoning (generate insights)
reasoning_prompt = PromptTemplate.from_template("""
You are an experienced AI analytics assistant.

Using the data and past insights provided in the "Relevant Data Context", generate 3–5 concise business insights related to the user's question.

Guidelines:
- Prioritize providing exact numerical values and specific timeframes when available in the context.
- Mimic the style and tone of the provided "Relevant Data Context" if it contains pre-written insights.
- NEVER include metrics, products, or timeframes not explicitly supported by the "Relevant Data Context".
- Focus on recent trends when comparing months; explicitly name the months and values.
- Reference actual metric values, product names, and timeframes.
- If the context is too sparse or inconclusive for the user's question, state this clearly instead of guessing or fabricating.
- Format the insights as a simple list, one insight per line.

User Question:
{question}

Relevant Data Context:
{retrieved_data}

Insights:
""")

# 8. Validation
validate_prompt = PromptTemplate.from_template("""
You are a validation agent tasked with auditing the insights generated by an AI assistant.

Instructions:
- For each insight, verify whether it is fully supported by the "Relevant Data Context".
- Check whether the values, trends, and comparisons are consistent with the "Relevant Data Context".
- Highlight any unsupported claims, hallucinations, or vague language.
- If needed, suggest revisions to the insight or suggest its removal.
- Flag any contradictory statements between the insights.

Insights to validate:
{insights}

Relevant Data Context:
{retrieved_data}

Return a detailed validation report with reasoning per insight. If all insights are valid, explicitly state that "All insights are valid and supported by the data."
""")

# 9. Final summary formatting
final_prompt = PromptTemplate.from_template("""
You are presenting the final output to a business user.

Summarize the validated insights as clear bullet points for an executive audience.

Formatting:
• Begin each bullet with a newline and a '•'
• Keep each insight concise and actionable
• Reference key metrics and values explicitly
• Maintain a professional, business-appropriate tone

Insights:
{insights}

Validation Notes:
{validation}

Final Answer:
""")

# --- Define the basic LLM chains for re-use inside the overall chain ---
# These are the basic prompt | llm chains. The debug wrappers will be added in create_overall_rag_chain.
reasoning_llm_chain = reasoning_prompt | llm
validate_llm_chain = validate_prompt | llm
final_llm_chain = final_prompt | llm


# --- Helper functions for debugging LLM inputs (used within overall chain) ---
def _debug_llm_input(inputs: Dict[str, Any], chain_name: str) -> Dict[str, Any]:
    """Generic debug wrapper for LLM chain inputs."""
    print(f"\nDEBUG: Input to {chain_name} (keys): {inputs.keys()}")
    # Print types of specific expected variables for LLM chains
    if chain_name == "Reasoning_Chain":
        print(f"DEBUG: {chain_name} input -> question type: {type(inputs.get('question'))}, retrieved_data type: {type(inputs.get('retrieved_data'))}")
    elif chain_name == "Validation_Chain":
        print(f"DEBUG: {chain_name} input -> insights type: {type(inputs.get('insights'))}, retrieved_data type: {type(inputs.get('retrieved_data'))}")
    elif chain_name == "Final_Formatting_Chain":
        print(f"DEBUG: {chain_name} input -> insights type: {type(inputs.get('insights'))}, validation type: {type(inputs.get('validation'))}")
    # print(f"DEBUG: Input to {chain_name} (full):\n{json.dumps(inputs, indent=2)}") # Uncomment for full debug
    return inputs


# --- Overall Chain (Orchestration) ---
# src/llm_interface.py

# ... (all existing code) ...

def create_overall_rag_chain(vectorstore):
    """
    Creates the complete RAG chain, binding the vector store for retrieval.
    """
    overall_chain = (
        RunnablePassthrough.assign(
            intent=intent_chain,
            question=lambda x: x["input"], # Original user question string
        )
        .assign(
            metadata=understand_chain.with_config(run_name="Extract_Metadata") | clean_metadata_chain,
        )
        .assign(
            retrieval_output=RunnableLambda(
                lambda inputs: retrieve_and_validate(inputs,  vectorstore)
            ).with_config(run_name="Retrieve_Data_from_VectorStore"),
        )
        .assign(
            processed_retrieval=lambda x: x["retrieval_output"],
            metric_check_output=RunnableLambda(check_metrics_and_suggest_on_docs),
            extracted_product_metric_context=lambda x: (
                f"Extracted Focus: Metrics: {x['metadata'].get('metrics', 'None')}, "
                f"Products: {x['metadata'].get('products', 'None')}\n"
                if x['metadata'].get('metrics') or x['metadata'].get('products') else ""
            )
        )
        .assign(
            insights=RunnableBranch(
                (
                    lambda x: "error" in x.get("processed_retrieval", {}) or "error" in x.get("metric_check_output", {}),
                    RunnableLambda(lambda x: x.get("processed_retrieval", {}).get("error", x.get("metric_check_output", {}).get("error", "An error occurred during retrieval or metric check.")))
                ),
                # --- THIS IS THE CRITICAL CHANGE ---
                RunnablePassthrough.assign(
                    retrieved_data=lambda x: x["processed_retrieval"]["retrieved_data"],
                    # Ensure 'question' is also passed if reasoning_prompt needs it separately from 'input'
                    # Although 'question' is already a top-level key, being explicit here ensures propagation.
                    question=lambda x: x["question"],
                    # Also pass extracted_product_metric_context to the chain, if reasoning_llm_chain needs it
                    extracted_product_metric_context=lambda x: x["extracted_product_metric_context"]
                )
                | RunnableLambda(lambda x: _debug_llm_input(x, "Reasoning_Chain"))
                | reasoning_llm_chain.with_config(run_name="Generate_Insights")
            ),
        )
        # --- You'll need to do a similar fix for 'validation' and 'final_answer' branches ---
        .assign(
            validation=RunnableBranch(
                (
                    lambda x: "error" in x.get("processed_retrieval", {}) or "error" in x.get("metric_check_output", {}),
                    RunnableLambda(lambda x: "Validation skipped due to prior error in retrieval or metric check.")
                ),
                # --- FIX FOR VALIDATION CHAIN ---
                RunnablePassthrough.assign(
                    insights=lambda x: x["insights"], # Pass the insights generated from the previous step
                    retrieved_data=lambda x: x["processed_retrieval"]["retrieved_data"] # Pass the retrieved data for validation
                )
                | RunnableLambda(lambda x: _debug_llm_input(x, "Validation_Chain"))
                | validate_llm_chain.with_config(run_name="Validate_Insights")
            ),
        )
        .assign(
            final_answer=RunnableBranch(
                (
                    lambda x: "error" in x.get("processed_retrieval", {}) or "error" in x.get("metric_check_output", {}),
                    RunnableLambda(lambda x: x.get("processed_retrieval", {}).get("error", x.get("metric_check_output", {}).get("error", "An error occurred during retrieval or metric check.")))
                ),
                # --- FIX FOR FINAL FORMATTING CHAIN ---
                RunnablePassthrough.assign(
                    insights=lambda x: x["insights"], # Insights from the reasoning step
                    validation=lambda x: x["validation"] # Validation notes from the validation step
                )
                | RunnableLambda(lambda x: _debug_llm_input(x, "Final_Formatting_Chain"))
                | final_llm_chain.with_config(run_name="Final_Formatting")
            )
        )
    )
    return overall_chain

# --- Helper function for external use (e.g., in main.py) ---
def query_rag_system(rag_chain, question):
    """
    Invokes the full RAG chain with a user question.
    """
    response = rag_chain.invoke({"input": question, "question": question})
    return response["final_answer"]


def extract_filters_from_query_llm(query):
    # This mock function is kept for backward compatibility with main.py's
    # initial structure, though the overall_chain handles this internally.
    # You can remove this function if main.py is fully refactored to just call
    # the overall_chain.
    mock_filters = {}
    if "Product A" in query:
        mock_filters["products"] = ["Product A"]
    if "Sales" in query:
        mock_filters["metrics"] = ["Sales"]
    if "March 2024" in query:
        mock_filters["time_point_prefix"] = "2024-03"
    if "value" in query or "number" in query:
        mock_filters["type"] = "metric_summary"
    if "trend" in query:
        mock_filters["type"] = "trend_summary"
        mock_filters["analysis_type"] = "trend"
    if "insight" in query or "overview" in query:
        mock_filters["type"] = "insight"
        mock_filters["analysis_type"] = "insight"
    return mock_filters


