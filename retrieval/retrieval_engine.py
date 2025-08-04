"""
Document Retrieval Engine

Handles document retrieval based on query objects with d                # Print all scores for debugging
                print(f"DEBUG - All distance scores for '{search_query}' (lower = more similar):")
                for i, (doc, score) in enumerate(docs_with_scores):
                    print(f"  Doc {i+1}: distance={score:.4f} - {doc.page_content[:100]}...")
                # Filter by distance threshold (keep documents with distance <= threshold)
                docs = [doc for doc, score in docs_with_scores if score <= score_threshold]
                print(f"Overall summary docs without product: {len(docs)} (filtered from {len(docs_with_scores)} with distance threshold <= {score_threshold})")t strategies 
for metric_summary vs overall_summary question types.
Vector search naturally handles unavailable products/metrics.
"""

from typing import List, Tuple
from langchain_core.documents import Document
from utils import build_chroma_filter, get_dynamic_k


def retrieve_documents(query_result, vectorstore, available_products, available_metrics, score_threshold: float = 1.0) -> Tuple[List[Document], List[Document]]:
    """
    Retrieve documents based on query objects with different strategies for metric_summary vs overall_summary.
    Vector search naturally handles unavailable products/metrics by returning fewer or no relevant documents.
    
    Args:
        query_result: List of QueryObject instances from the query parser
        vectorstore: Chroma vectorstore instance
        available_products: List of available product names
        available_metrics: List of available metric names
        score_threshold: Maximum distance threshold for documents to be included (default: 1.0)
                        Lower values = stricter filtering, higher values = more lenient
                        Uses distance metrics where LOWER scores = MORE similar
    
    Returns:
        tuple: (metric_query_docs, overall_summary_docs)
            - metric_query_docs: Documents retrieved for metric-specific queries
            - overall_summary_docs: Documents retrieved for overall summary queries
    """
    metric_query_docs = []
    overall_summary_docs = []
    
    # Process all query objects - let vector search handle availability naturally
    for obj in query_result:
        print(f"Processing query object: product={obj.product}, metric={obj.metric}, question_type={obj.question_type}")
        
        if obj.question_type == "metric_summary":
            # For metric_summary: search by product, metrics, and time_spans
            chroma_filter = build_chroma_filter(obj.product, [], obj.time_spans)
            print(f"Metric summary filter for {obj.product}, {obj.metric}, {obj.time_spans}: {chroma_filter}")
            k = get_dynamic_k([obj])  # Pass the single object as a list
            search_query = f"{obj.product} {obj.metric}"
            docs_with_scores = vectorstore.similarity_search_with_score(query=search_query, k=k, filter=chroma_filter)
            # Print all scores for debugging
            print(f"DEBUG - All distance scores for '{search_query}' (lower = more similar):")
            for i, (doc, score) in enumerate(docs_with_scores):
                print(f"  Doc {i+1}: distance={score:.4f} - {doc.page_content[:100]}...")
            # Filter by distance threshold (keep documents with distance <= threshold)
            docs = [doc for doc, score in docs_with_scores if score <= score_threshold]
            print(f"Metric summary docs for {obj.product}, {obj.metric}: {len(docs)} (filtered from {len(docs_with_scores)} with distance threshold <= {score_threshold})")
            metric_query_docs.extend(docs)
            
        elif obj.question_type == "overall_summary":
            # For overall_summary: different logic based on product availability
            if obj.product and obj.product.strip():
                # Product is specified: search by product and time_spans
                chroma_filter = build_chroma_filter(obj.product, [], obj.time_spans)
                print(f"Overall summary filter with product {obj.product}, {obj.time_spans}: {chroma_filter}")
                k = get_dynamic_k([obj])  # Pass the single object as a list
                k = k * len(available_metrics)  # Adjust k for multiple metrics
                search_query = obj.product
                docs_with_scores = vectorstore.similarity_search_with_score(query=search_query, k=k, filter=chroma_filter)
                # Print all scores for debugging
                print(f"DEBUG - All distance scores for '{search_query}' (lower = more similar):")
                for i, (doc, score) in enumerate(docs_with_scores):
                    print(f"  Doc {i+1}: distance={score:.4f} - {doc.page_content[:100]}...")
                # Filter by distance threshold (keep documents with distance <= threshold)
                docs = [doc for doc, score in docs_with_scores if score <= score_threshold]
                print(f"Overall summary docs with product {obj.product}: {len(docs)} (filtered from {len(docs_with_scores)} with distance threshold <= {score_threshold})")
            else:
                # Product is not specified: search by time_spans only
                chroma_filter = build_chroma_filter("", [], obj.time_spans)
                print(f"Overall summary filter without product, {obj.time_spans}: {chroma_filter}")
                # Create a synthetic query object representing all products and metrics
                synthetic_objects = []
                for product in available_products:
                    for metric in available_metrics:
                        synthetic_obj = type('obj', (), {
                            'product': product,
                            'metric': metric,
                            'time_spans': obj.time_spans
                        })()
                        synthetic_objects.append(synthetic_obj)
                k = get_dynamic_k(synthetic_objects)
                search_query = "summary overview"
                docs_with_scores = vectorstore.similarity_search_with_score(query=search_query, k=k, filter=chroma_filter)
                # Print all scores for debugging
                print(f"DEBUG - All similarity scores for '{search_query}':")
                for i, (doc, score) in enumerate(docs_with_scores):
                    print(f"  Doc {i+1}: score={score:.4f} - {doc.page_content[:100]}...")
                # Filter by score threshold
                docs = [doc for doc, score in docs_with_scores if score >= score_threshold]
                print(f"Overall summary docs without product: {len(docs)} (filtered from {len(docs_with_scores)} with threshold {score_threshold})")
            overall_summary_docs.extend(docs)
        
        print(f"Retrieved {len(docs)} docs for this query object")
    
    print(f"Total retrieved docs - Metric queries: {len(metric_query_docs)}, Overall summary: {len(overall_summary_docs)}")
    return metric_query_docs, overall_summary_docs


def retrieve_insight_docs(vectorstore, k: int = 10) -> List[Document]:
    """
    Retrieve insight documents from the vectorstore.
    
    Args:
        vectorstore: Chroma vectorstore instance
        k: Number of insight documents to retrieve
    
    Returns:
        List[Document]: List of insight documents
    """
    insight_docs = vectorstore.similarity_search(
        query="insight",
        k=k,
        filter={"type": "insight"}
    )
    print(f"Retrieved {len(insight_docs)} insight documents")
    return insight_docs
