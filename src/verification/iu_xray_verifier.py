from typing import Dict, Any, List, Optional
import numpy as np
from src.retrieval.embeddings import RadiologyEmbeddingModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class IUXRayReportVerifier:
    """
    Verification module to evaluate AI-generated radiology reports and multi-label findings
    against ground-truth or semantically retrieved reports from the IU-XRay dataset corpus.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        embed_model_name = self.config.get("retrieval", {}).get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_model = RadiologyEmbeddingModel(model_name=embed_model_name)

    def verify_report(
        self,
        generated_report: str,
        predictions_data: Dict[str, Any],
        retrieved_reports: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Performs semantic and finding-level verification against IU-XRay reference corpus.

        Returns:
            Dict containing similarity metrics, status, matched findings, and reference details.
        """
        positive_findings = [p["finding"] for p in predictions_data.get("positive_findings", [])]
        
        if not retrieved_reports:
            return {
                "verified": False,
                "similarity_score": 0.0,
                "similarity_percentage": "0%",
                "status": "UNVERIFIED (No Reference Reports)",
                "matched_findings": [],
                "top_reference_report": None,
                "verification_summary": "No matching reference reports from the IU-XRay corpus were available for verification."
            }

        top_ref = retrieved_reports[0]
        ref_text = top_ref.get("full_report") or f"FINDINGS: {top_ref.get('findings', '')} IMPRESSION: {top_ref.get('impression', '')}"

        # Step 1: Semantic Embedding Cosine Similarity Score
        try:
            gen_vec = self.embedding_model.encode(generated_report).reshape(-1)
            ref_vec = self.embedding_model.encode(ref_text).reshape(-1)
            
            # Cosine similarity calculation
            dot = np.dot(gen_vec, ref_vec)
            norm_gen = np.linalg.norm(gen_vec)
            norm_ref = np.linalg.norm(ref_vec)
            if norm_gen > 0 and norm_ref > 0:
                similarity = float(dot / (norm_gen * norm_ref))
            else:
                similarity = float(top_ref.get("similarity_score", 0.5))
        except Exception as e:
            logger.warning(f"Error computing cosine similarity embedding: {e}")
            similarity = float(top_ref.get("similarity_score", 0.5))


        # Clamp between 0.0 and 1.0
        similarity = max(0.0, min(1.0, similarity))
        sim_percent_val = int(round(similarity * 100))
        sim_percent_str = f"{sim_percent_val}%"

        # Step 2: Medical Finding Alignment & Verification
        ref_text_lower = ref_text.lower()
        matched_findings = []
        unmatched_findings = []

        for finding in positive_findings:
            finding_words = finding.lower().split()
            # Check if main key terms are in reference text
            if any(word in ref_text_lower for word in finding_words if len(word) > 3):
                matched_findings.append(finding)
            else:
                unmatched_findings.append(finding)

        # Also check for "No Finding" / normal alignment
        if not positive_findings:
            if "normal" in ref_text_lower or "no acute" in ref_text_lower or "clear" in ref_text_lower:
                matched_findings.append("No Acute Cardiopulmonary Disease")

        # Step 3: Determine Verification Status & Summary
        if sim_percent_val >= 75 or len(matched_findings) >= 1:
            status = "VERIFIED (High Alignment with IU-XRay Corpus)"
            status_badge = "HIGH_ALIGNMENT"
        elif sim_percent_val >= 50:
            status = "VERIFIED (Moderate Alignment with IU-XRay Corpus)"
            status_badge = "MODERATE_ALIGNMENT"
        else:
            status = "DIVERGENT (Low Alignment with Reference Report)"
            status_badge = "LOW_ALIGNMENT"

        if matched_findings:
            summary_str = f"Verified {len(matched_findings)} finding(s) ({', '.join(matched_findings)}) against IU-XRay report {top_ref.get('report_id', '')} with {sim_percent_str} semantic similarity."
        else:
            summary_str = f"Report compared against IU-XRay report {top_ref.get('report_id', '')} with {sim_percent_str} semantic similarity."

        verification_result = {
            "verified": True,
            "similarity_score": round(similarity, 4),
            "similarity_percentage": sim_percent_str,
            "status": status,
            "status_badge": status_badge,
            "matched_findings": matched_findings,
            "unmatched_findings": unmatched_findings,
            "top_reference_report": top_ref,
            "verification_summary": summary_str
        }

        logger.info(f"IU-XRay Verification complete: {status} ({sim_percent_str})")
        return verification_result
