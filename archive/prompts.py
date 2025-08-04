from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnableBranch
import json
import pandas as pd
import sqlite3
from difflib import get_close_matches
import re

llm = OllamaLLM(model="llama3")

# --- Intent classification ---
intent_prompt = PromptTemplate.from_template("""
Classify the following user input as either "question" or "feedback". Respond with ONLY one of these words, and nothing else.

If the input contains a question about data or analysis, classify as "question".
If the input is a reaction, suggestion, or comment about the assistant's output, classify as "feedback".

Input: {input}
Output:
""")
intent_chain = intent_prompt | llm

# --- Understanding the question ---
understand_prompt = PromptTemplate.from_template("""
You are an intelligent assistant helping analyze business questions.

Your job is to extract the following from the user's question:
- metrics: The key metric(s) mentioned (as a list of strings), or null if none
- products: The product(s) mentioned (as a list of strings), or null if none
- time_range: The specific time range (e.g., "Jan 2025", "Q1 2024", or null)
- analysis_type: One of [summary, trend, anomaly, comparison], or null

Return your output as **strictly valid JSON** with exactly these keys. Do not include any explanation, markdown, or formatting.

User Question:
{question}

Output:
""")

understand_chain = understand_prompt | llm

# --- Clean metadata ---
def clean_metadata(llm_output):
    try:
        data = json.loads(llm_output)
    except Exception:
        return {"error": "Invalid JSON format"}

    cleaned = {}
    for k, v in data.items():
        key = k.strip().lower()
        if isinstance(v, str):
            cleaned[key] = v.strip()
        elif isinstance(v, list):
            cleaned[key] = [item.strip() for item in v]
        else:
            cleaned[key] = v
    return cleaned

clean_metadata_chain = RunnableLambda(clean_metadata)

# --- Retrieve relevant data ---
def fetch_relevant_data(meta):
    metrics = meta.get("metrics", [])
    products = meta.get("products", [])
    time_range = meta.get("time_range", "")

    if not metrics or not products:
        return {"retrieved_data": [], "meta": meta, "error": "Missing metric/product in metadata"}

    conn = sqlite3.connect("your_data.db")
    query = f"""
        SELECT *
        FROM metrics_table
        WHERE metric IN ({','.join(['?']*len(metrics))})
        AND product IN ({','.join(['?']*len(products))})
    """
    params = metrics + products
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    return {
        "retrieved_data": df.to_dict(orient="records"),
        "meta": meta
    }

retrieve_chain = RunnableLambda(fetch_relevant_data)

# --- Check for exact or fuzzy metric match ---
def check_metrics_and_suggest(inputs):
    metadata = inputs.get("meta", {})
    requested = metadata.get("metrics", [])
    retrieved_data = inputs.get("retrieved_data", [])

    available_metrics = {row["metric"] for row in retrieved_data if "metric" in row}
    available_lower = {m.lower() for m in available_metrics}
    requested_lower = [m.lower() for m in requested]

    missing = [m for m in requested_lower if m not in available_lower]

    if not missing:
        return inputs  # ✅ everything matches

    # Fuzzy match suggestions
    suggestions = {}
    for m in missing:
        match = get_close_matches(m, available_lower, n=1, cutoff=0.6)
        if match:
            original_requested = next(orig for orig in requested if orig.lower() == m)
            suggestions[original_requested] = next(orig for orig in available_metrics if orig.lower() == match[0])

    return {
        "error": (
            f"The following metrics are not in the dataset: {', '.join(missing)}. "
            f"Closest matches found: {suggestions}. No insights were generated."
        ),
        "suggestions": suggestions,
        "meta": metadata,
        "retrieved_data": retrieved_data
    }

check_metrics_chain = RunnableLambda(check_metrics_and_suggest)

# --- Reasoning (generate insights) ---
reasoning_prompt = PromptTemplate.from_template("""
You are an experienced AI analytics assistant.

Using the data and past insights below, generate 3–5 concise business insights.

Guidelines:
- Mimic the style and tone of the past insights.
- NEVER include metrics not in the retrieved data, even if they appear in past insights.
- Focus on recent trends when comparing months; explicitly name the months and values.
- Reference actual metric values, product names, and timeframes.
- If the data is too sparse or inconclusive, state this instead of guessing.

User Question:
{question}

Relevant Data:
{retrieved_data}

Past Insights:
{past_insights}

Insights:
""")

reasoning_chain = reasoning_prompt | llm

# --- Validation ---
validate_prompt = PromptTemplate.from_template("""
You are a validation agent tasked with auditing the insights generated by an AI assistant.

Instructions:
- For each insight, verify whether it is fully supported by the data.
- Check whether the values, trends, and comparisons are consistent with the retrieved data.
- Highlight any unsupported claims, hallucinations, or vague language.
- If needed, revise the insight or suggest its removal.
- Flag any contradictory statements between the insights.

Insights to validate:
{insights}

Relevant Data:
{retrieved_data}

Return a detailed validation report with reasoning per insight. If all insights are valid, explicitly state that.
""")

validate_chain = validate_prompt | llm

# --- Final summary formatting ---
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

final_chain = final_prompt | llm
