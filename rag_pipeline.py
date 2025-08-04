"""
RAG Pipeline Module

Handles the complete RAG (Retrieval-Augmented Generation) pipeline
including intent classification, query parsing, document retrieval, 
reasoning, and final formatting.
"""

from utils import get_latest_month_year_str, replace_latest_in_time_spans
from retrieval.retrieval_engine import retrieve_documents
from chain.intent_classification.intent_classifier import IntentClassifier
from chain.query_parsing.query_parser import QueryParser
from chain.answer_generation.answer_generator import AnswerGenerator
from chain.final_formatting.final_formatter import FinalFormatter


class RAGPipeline:
    """
    Main RAG pipeline class that orchestrates the complete workflow
    from user input to final formatted response.
    """
    
    def __init__(self, vectorstore, available_products, available_metrics, score_threshold: float = 0.3):
        """
        Initialize the RAG pipeline with required components.
        
        Args:
            vectorstore: Chroma vectorstore instance
            available_products: List of available product names
            available_metrics: List of available metric names
            score_threshold: Minimum similarity score for document retrieval (default: 0.3)
        """
        self.vectorstore = vectorstore
        self.available_products = available_products
        self.available_metrics = available_metrics
        self.score_threshold = score_threshold
        
        # Initialize chains
        self.intent_chain = IntentClassifier()
        self.query_chain = QueryParser()
        self.reasoning_chain = AnswerGenerator(model_name="llama3")
        self.final_formatter = FinalFormatter(model_name="llama3")
        
        # Get format instructions
        self.format_instructions = self.query_chain.format_instructions
    
    def get_rag_response(self, user_input: str) -> str:
        """
        Process user input through the complete RAG pipeline.
        
        Args:
            user_input: User's question or input
            
        Returns:
            str: Final formatted response
        """
        # Step 1: Classify intent (question or feedback)
        intent_result = self.intent_chain.run({"question": user_input})
        
        if intent_result.get("input_type") == "feedback":
            return "Noted. Thanks for the feedback!"
        
        if intent_result.get("input_type") == "unsupported":
            return "Sorry, I am unable to answer that type of question."
        
        # Step 2: Parse query for products, metrics, and time spans
        query_result = self.query_chain.run({
            "question": user_input,
            "intent": intent_result,
            "available_products": self.available_products,
            "available_metrics": self.available_metrics,
            "format_instructions": self.format_instructions
        })

        # Step 3: Replace 'latest' in all time_spans for each query object
        latest_month = get_latest_month_year_str(self.vectorstore)
        for obj in query_result:
            replace_latest_in_time_spans(obj.time_spans, latest_month)
        print("Query result:", query_result, '\n')  # Debugging output

        # Step 4: Retrieve documents using the retrieval engine
        metric_query_docs, overall_summary_docs = retrieve_documents(
            query_result, self.vectorstore, self.available_products, self.available_metrics, self.score_threshold
        )
        
        print(f"DEBUG - Retrieved docs: metric={len(metric_query_docs)}, overall={len(overall_summary_docs)}")

        # Continue with normal processing - let vector search handle availability naturally
        print("DEBUG - Processing with available documents")
        return self._handle_normal_processing(user_input, metric_query_docs, overall_summary_docs)
    
    def _handle_normal_processing(self, user_input, metric_query_docs, overall_summary_docs):
        """
        Handle normal processing when documents are available.
        Vector search naturally handles unavailable items by returning fewer/no docs.
        """
        all_docs = metric_query_docs + overall_summary_docs
        
        # Check if we got any relevant documents
        if not all_docs:
            return "No relevant information found for your request. Please check if the products and metrics you mentioned are available in the data."
        
        # Step 5: Aggregate context and reasoning
        metric_docs = [doc for doc in all_docs if doc.metadata.get("type") == "metric_summary"]
        insight_docs = []
        # insight_docs = retrieve_insight_docs(self.vectorstore)

        context = ""
        if insight_docs:
            context += "PAST INSIGHTS:\n" + "\n".join(doc.page_content for doc in insight_docs) + "\n"
        if metric_docs:
            context += "METRIC SUMMARIES:\n" + "\n".join(doc.page_content for doc in metric_docs)
        
        # Print out metric summaries for debugging
        print("\n[DEBUG] Metric Summaries:")
        for doc in metric_docs:
            print(doc.page_content)
        
        answer = self.reasoning_chain.run(user_input, context)
        print("\nLLM Answer:\n", answer)
        
        # Step 6: Apply final formatting
        formatted_answer = self.final_formatter.run(answer, user_input)
        print("\nFormatted Answer:\n", formatted_answer)
        
        return formatted_answer
