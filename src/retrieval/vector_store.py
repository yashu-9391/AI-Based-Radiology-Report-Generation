import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class FAISSVectorStore:
    """
    FAISS-based vector database for indexing and retrieving radiology reports.
    Includes serialization to disk and similarity search.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._init_faiss()

    def _init_faiss(self):
        try:
            import faiss
            # IndexFlatIP with normalized L2 vectors equals Cosine Similarity
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("Initialized FAISS IndexFlatIP vector store.")
        except Exception as e:
            logger.warning(f"FAISS library initialization notice: {e}. Using fallback NumPy Cosine Similarity engine.")
            self.index = None
            self.vectors = None

    def add_vectors(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """
        Adds normalized vector embeddings and corresponding document metadata.
        """
        if len(embeddings) == 0:
            return

        # L2 Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_embeddings = (embeddings / norms).astype(np.float32)

        if self.index is not None:
            self.index.add(normalized_embeddings)
        
        if not hasattr(self, 'vectors') or self.vectors is None or len(self.vectors) == 0:
            self.vectors = normalized_embeddings
        else:
            self.vectors = np.vstack([self.vectors, normalized_embeddings])

        self.metadata.extend(metadata_list)
        logger.info(f"Added {len(embeddings)} vectors to index. Total count: {len(self.metadata)}")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search and returns top-K nearest radiology reports.
        """
        if len(self.metadata) == 0:
            logger.warning("Vector store is empty. No reports to retrieve.")
            return []

        # Ensure 2D query
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        query_normalized = (query_embedding / norm).astype(np.float32)

        k = min(top_k, len(self.metadata))

        if self.index is not None and getattr(self.index, 'ntotal', 0) > 0:
            scores, indices = self.index.search(query_normalized, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata) and idx >= 0:
                    meta = self.metadata[idx].copy()
                    meta["similarity_score"] = float(score)
                    results.append(meta)
            return results
        elif hasattr(self, 'vectors') and self.vectors is not None and len(self.vectors) > 0:
            # Fallback NumPy Cosine Search
            sims = np.dot(self.vectors, query_normalized.T).squeeze(1)
            top_indices = np.argsort(sims)[::-1][:k]
            results = []
            for idx in top_indices:
                meta = self.metadata[idx].copy()
                meta["similarity_score"] = float(sims[idx])
                results.append(meta)
            return results
        else:
            logger.warning("Vector store has no active FAISS index or vector array. Returning empty results.")
            return []

    def save(self, index_path: str, metadata_path: str):
        """
        Saves vector index and report metadata to disk.
        """
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        if self.index is not None:
            import faiss
            faiss.write_index(self.index, index_path)
        
        payload = {
            "metadata": self.metadata,
            "vectors": self.vectors if hasattr(self, 'vectors') and self.vectors is not None else None
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(payload, f)

        logger.info(f"Saved vector index to {index_path} and metadata to {metadata_path}")

    def load(self, index_path: str, metadata_path: str) -> bool:
        """
        Loads vector index and report metadata from disk.
        """
        if not os.path.exists(metadata_path):
            logger.warning(f"Vector store metadata file ({metadata_path}) not found.")
            return False

        try:
            if os.path.exists(index_path):
                try:
                    import faiss
                    loaded_index = faiss.read_index(index_path)
                    if loaded_index is not None and loaded_index.ntotal > 0:
                        self.index = loaded_index
                except Exception as fe:
                    logger.warning(f"FAISS index load notice ({fe}). Using fallback NumPy vectors.")
            
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    self.metadata = data.get("metadata", [])
                    self.vectors = data.get("vectors", None)
                elif isinstance(data, list):
                    self.metadata = data

            logger.info(f"Successfully loaded vector store with {len(self.metadata)} records.")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False

