import os
from typing import Dict, Any, List, Optional
from src.retrieval.embeddings import RadiologyEmbeddingModel
from src.retrieval.vector_store import FAISSVectorStore
from src.data.iu_xray_dataset import IUXRayDatasetLoader
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Baseline reference report corpus for IU X-Ray fallback
DEFAULT_REFERENCE_REPORTS = [
    {
        "report_id": "REF_001",
        "findings": "Heart size is moderately enlarged. Bilateral pleural effusions noted with mild bibasilar interstitial edema.",
        "impression": "Cardiomegaly with bilateral pleural effusion and mild pulmonary edema."
    },
    {
        "report_id": "REF_002",
        "findings": "The lungs are clear without focal consolidation, pneumothorax, or pleural effusion. Lungs are well aerated.",
        "impression": "No acute cardiopulmonary disease."
    },
    {
        "report_id": "REF_003",
        "findings": "Focal opacity in the right lower lobe with ill-defined margins. No pleural effusion or pneumothorax.",
        "impression": "Right lower lobe pneumonia / consolidation."
    },
    {
        "report_id": "REF_004",
        "findings": "Mild linear opacity at the left base indicative of discoid atelectasis. Cardiac silhouette is within normal limits.",
        "impression": "Minor basilar atelectasis."
    },
    {
        "report_id": "REF_005",
        "findings": "Prominent pulmonary vascularity with diffuses bilateral alveolar infiltrates and Kerley B lines.",
        "impression": "Moderate pulmonary edema secondary to congestive heart failure."
    }
]

class ReportRetriever:
    """
    High-level report retrieval manager connecting embedding encoding with FAISS indexing.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        retrieval_cfg = config.get("retrieval", {})
        embed_model_name = retrieval_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        
        self.embedding_model = RadiologyEmbeddingModel(model_name=embed_model_name)
        self.vector_store = FAISSVectorStore(dimension=384)
        
        self.index_path = retrieval_cfg.get("vector_db_path", "checkpoints/faiss_index.bin")
        self.metadata_path = retrieval_cfg.get("corpus_metadata_path", "checkpoints/report_corpus.pkl")
        self.top_k = retrieval_cfg.get("top_k", 3)

        # Attempt auto-loading existing index
        self.is_indexed = self.vector_store.load(self.index_path, self.metadata_path)
        if not self.is_indexed or len(self.vector_store.metadata) < 10:
            self._build_default_index()

    def _build_default_index(self):
        """
        Builds initial vector index from IU X-Ray dataset if present, or defaults to reference corpus.
        """
        iu_path = self.config.get("data", {}).get("iu_xray_path", "datasets/iu_xray")
        if os.path.exists(iu_path):
            try:
                count = self.build_index_from_dataset(iu_path)
                if count > 10:
                    logger.info(f"Successfully auto-indexed {count} reports from IU X-Ray dataset into FAISS.")
                    return
            except Exception as e:
                logger.warning(f"Auto-indexing IU X-Ray dataset failed ({e}). Falling back to baseline corpus.")

        logger.info("Building vector index from default reference radiology report corpus...")
        texts = [f"FINDINGS: {r['findings']} IMPRESSION: {r['impression']}" for r in DEFAULT_REFERENCE_REPORTS]
        embeddings = self.embedding_model.encode(texts)
        self.vector_store.add_vectors(embeddings, DEFAULT_REFERENCE_REPORTS)
        self.vector_store.save(self.index_path, self.metadata_path)
        self.is_indexed = True


    def build_index_from_dataset(self, iu_xray_path: Optional[str] = None) -> int:
        """
        Processes IU X-Ray dataset reports and indexes them into the vector database.
        """
        path = iu_xray_path or self.config.get("data", {}).get("iu_xray_path", "datasets/iu_xray")
        dataset = IUXRayDatasetLoader(root_dir=path, config=self.config)
        reports = dataset.get_reports()

        if not reports:
            logger.warning("No reports loaded from IU X-Ray dataset. Retaining reference index.")
            return len(self.vector_store.metadata)

        logger.info(f"Indexing {len(reports)} radiology reports into FAISS...")
        texts = [r["full_report"] for r in reports]
        embeddings = self.embedding_model.encode(texts)

        # Reset store & index new corpus
        self.vector_store = FAISSVectorStore(dimension=384)
        self.vector_store.add_vectors(embeddings, reports)
        self.vector_store.save(self.index_path, self.metadata_path)
        self.is_indexed = True
        return len(reports)

    def search_similar_reports(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Encodes query text and searches vector store for top-K semantically similar reports.
        """
        k = top_k or self.top_k
        if not query_text.strip():
            logger.warning("Query text is empty.")
            return []

        query_vec = self.embedding_model.encode(query_text)
        results = self.vector_store.search(query_vec, top_k=k)
        return results
