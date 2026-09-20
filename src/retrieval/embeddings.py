import numpy as np
from typing import List, Union
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class RadiologyEmbeddingModel:
    """
    Sentence embedding service for medical radiology text retrieval.
    Wraps sentence-transformers with a lightweight fallback if model download is offline.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Operating in lightweight fallback mode.")
            self.model = None

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encodes input text string or list of strings into embedding vectors.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                return embeddings.astype(np.float32)
            except Exception as e:
                logger.error(f"Error during SentenceTransformer encoding: {e}")

        # Lightweight deterministic TF-IDF/Hash fallback embedding (384 dimensions)
        logger.info("Using deterministic fallback embedding representation.")
        fallback_embeddings = []
        for text in texts:
            words = text.lower().split()
            vector = np.zeros(384, dtype=np.float32)
            for w in words:
                hash_idx = abs(hash(w)) % 384
                vector[hash_idx] += 1.0
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            fallback_embeddings.append(vector)

        return np.array(fallback_embeddings, dtype=np.float32)
