from typing import Dict, Any, List
from src.rag.query_builder import build_retrieval_query
from src.rag.context_builder import build_rag_context
from src.retrieval.retriever import ReportRetriever
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class SingleAgentRAGPipeline:
    """
    Single-Agent Retrieval-Augmented Generation (RAG) orchestration pipeline.
    Orchestrates query formation, report retrieval, and grounded context construction.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retriever = ReportRetriever(config=config)

    def process(self, predictions_data: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
        """
        Executes single-agent RAG workflow.
        """
        logger.info("Executing Single-Agent RAG workflow...")
        
        # Step 1: Formulate retrieval query from multi-label predictions
        query_text = build_retrieval_query(predictions_data)
        logger.info(f"Generated retrieval query: '{query_text}'")

        # Step 2: Retrieve Top-K semantically similar radiology reports from vector store
        retrieved_reports = self.retriever.search_similar_reports(query_text=query_text, top_k=top_k)
        logger.info(f"Retrieved {len(retrieved_reports)} reference reports.")

        # Step 3: Build grounded context payload
        rag_context = build_rag_context(predictions_data, retrieved_reports)

        return {
            "query_text": query_text,
            "retrieved_reports": retrieved_reports,
            "rag_context": rag_context
        }
