from chain.intent_classification.intent_classifier import IntentClassifier
from chain.query_parsing.query_parser import QueryParser
import re
import json5
from json_utils import extract_json_from_text

# Initialize chain classes once
intent_chain = IntentClassifier()
query_chain = QueryParser()

def analyze_question(question: str):
    # Step 1: Classify as question or feedback
    intent_result = intent_chain.run(question)
    input_type = intent_result.get("input_type")

    if input_type == "feedback":
        print("response: Noted. Thanks for the feedback!")
        return None  # or raise or return special type

    # Step 2: Run metadata extraction (with context-based classification)
    metadata_result = query_chain.run(question)
    return metadata_result

from typing import List
from langchain_core.documents import Document

def filter_documents_by_metadata(docs: List[Document], metadata: dict) -> List[Document]:
    products = metadata.get("products", [])
    metrics = metadata.get("metrics", [])
    time_spans = metadata.get("time_spans", [])

    filtered = []

    for doc in docs:
        meta = doc.metadata

        product_ok = not products or meta.get("product") in products
        metric_ok = not metrics or meta.get("metric") in metrics

        time_ok = False
        doc_time = meta.get("month_year_str")
        if doc_time and time_spans:
            for span in time_spans:
                start = span.get("start")
                end = span.get("end")
                if start <= doc_time <= end:
                    time_ok = True
                    break
        else:
            time_ok = True  # No time filter

        if product_ok and metric_ok and time_ok:
            filtered.append(doc)

    return filtered



def build_chroma_filter(product, metric, time_spans, time_field="month_year_int"):
    """
    Build a Chroma filter for a single product/metric and a list of time_spans.
    product: str
    metric: str
    time_spans: list of TimeSpan or dict
    Returns a filter dict for Chroma.
    """
    def ym_str_to_int(ym):
        if ym is None:
            return None
        ym_str = str(ym)
        return int(ym_str.replace("-", ""))

    filters = []
    if product:
        filters.append({"product": product})
    if metric:
        filters.append({"metric": metric})
    if time_spans:
        time_filters = []
        for span in time_spans:
            start = getattr(span, "start", None) if not isinstance(span, dict) else span.get("start")
            end = getattr(span, "end", None) if not isinstance(span, dict) else span.get("end")
            start_int = ym_str_to_int(start)
            end_int = ym_str_to_int(end)
            if start_int and end_int:
                time_filters.append({
                    "$and": [
                        {time_field: {"$gte": start_int}},
                        {time_field: {"$lte": end_int}}
                    ]
                })
        if time_filters:
            filters.append({"$or": time_filters} if len(time_filters) > 1 else time_filters[0])
    if not filters:
        return {}
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}

def extract_products_and_metrics(vectorstore):
    products_set = set()
    metrics_set = set()
    metadatas = vectorstore.get()['metadatas']
    for meta in metadatas:
        if 'product' in meta:
            products_set.add(meta['product'])
        if 'metric' in meta:
            metrics_set.add(meta['metric'])
    return sorted(products_set), sorted(metrics_set)

def get_latest_month_year_str(vectorstore):
    all_metas = vectorstore.get()['metadatas']
    all_months = [meta.get('month_year_int') for meta in all_metas if meta.get('month_year_int') is not None]
    if not all_months:
        return None
    latest_int = max(all_months)
    # Convert int YYYYMM to string YYYY-MM
    latest_str = f"{latest_int // 100}-{latest_int % 100:02d}"
    return latest_str

def replace_latest_in_time_spans(time_spans, latest_month):
    """
    Replace 'latest' in the 'start' or 'end' fields of time span dicts/objects with the provided latest_month value.
    Mutates the input list in place.
    """
    for span in time_spans:
        # Handle dict or object with 'start' and 'end' attributes
        if getattr(span, "start", None) == "latest" or (isinstance(span, dict) and span.get("start") == "latest"):
            if isinstance(span, dict):
                span["start"] = latest_month
            else:
                span.start = latest_month
        if getattr(span, "end", None) == "latest" or (isinstance(span, dict) and span.get("end") == "latest"):
            if isinstance(span, dict):
                span["end"] = latest_month
            else:
                span.end = latest_month

def get_dynamic_k(query_objects):
    """
    Returns k for similarity search:
    - k = number of unique (product, metric) pairs * total months in all time spans.
    query_objects: list of dicts or objects with 'product', 'metric', 'time_spans'.
    """
    def count_months(start, end):
        if not start or not end:
            return 1
        start_year, start_month = map(int, start.split('-'))
        end_year, end_month = map(int, end.split('-'))
        return (end_year - start_year) * 12 + (end_month - start_month) + 1

    pairs = set()
    total_months = 0
    for obj in query_objects:
        product = getattr(obj, 'product', None) if not isinstance(obj, dict) else obj.get('product')
        metric = getattr(obj, 'metric', None) if not isinstance(obj, dict) else obj.get('metric')
        pairs.add((product, metric))
        time_spans = getattr(obj, 'time_spans', []) if not isinstance(obj, dict) else obj.get('time_spans', [])
        for span in time_spans:
            s = getattr(span, 'start', None) if not isinstance(span, dict) else span.get('start')
            e = getattr(span, 'end', None) if not isinstance(span, dict) else span.get('end')
            total_months += count_months(s, e)
    if len(pairs) == 1 and total_months == 1:
        return 1
    return max(1, len(pairs) * max(1, total_months))

def format_bullet_points(answer: str) -> str:
    answer = answer.replace("\\n", "\n")
    # Normalize bullets
    parts = re.split(r"[\n\r]+|•", answer)
    lines = []
    for part in parts:
        part = part.strip()
        if part:
            # Remove all line breaks and excessive spaces within the bullet
            part = re.sub(r"[\n\r]+", " ", part)
            part = re.sub(r"\s+", " ", part)
            lines.append(f"• {part}")
    # Add an extra newline between bullets for visual separation
    return "\n\n".join(lines)
