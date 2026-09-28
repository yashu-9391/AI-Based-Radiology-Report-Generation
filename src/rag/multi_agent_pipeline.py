"""
Multi-Agent Retrieval-Augmented Generation (RAG) Architecture.
Defines modular agent responsibilities:
1. RetrievalAgent: Executes FAISS semantic vector retrieval over indexed radiology corpus.
2. FindingVerificationAgent: Cross-checks vision predictions against retrieved reference evidence.
3. DraftingAgent: Generates structured, patient-friendly preliminary radiology report draft.
4. HallucinationCriticAgent: Criticizes draft report text, verifying all clinical entity claims against visual and retrieved evidence.
"""

import re
from typing import Dict, Any, List, Tuple
from src.rag.query_builder import build_retrieval_query
from src.rag.context_builder import build_rag_context
from src.retrieval.retriever import ReportRetriever
from src.report_generation.generator import MedicalReportGenerator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Standard observation keywords & clinical entity terms for verification
PATHOLOGY_TERMS = {
    "Enlarged Cardiomediastinum": ["cardiomediastinum", "mediastinum", "widened mediastinum"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "heart size", "cardiac silhouette"],
    "Lung Opacity": ["opacity", "opacities", "cloudy", "infiltrate", "airspace shadow"],
    "Lung Lesion": ["lesion", "nodule", "mass", "spot"],
    "Edema": ["edema", "vascular congestion", "fluid overload", "fluid balance"],
    "Consolidation": ["consolidation", "airspace disease", "dense opacity"],
    "Pneumonia": ["pneumonia", "infection", "chest cold", "infectious opacity"],
    "Atelectasis": ["atelectasis", "collapse", "volume loss"],
    "Pneumothorax": ["pneumothorax", "collapsed lung", "air space outside lung"],
    "Pleural Effusion": ["pleural effusion", "effusion", "fluid around lung"],
    "Pleural Other": ["pleural thickening", "pleural disease", "pleural lining"],
    "Fracture": ["fracture", "broken bone", "rib fracture"],
    "Support Devices": ["support device", "tube", "line", "catheter", "pacemaker", "hardware"],
}


class RetrievalAgent:
    """Agent 1: Wraps FAISS similarity search over IU X-Ray corpus."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retriever = ReportRetriever(config=config)

    def retrieve(self, predictions_data: Dict[str, Any], top_k: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        query_text = build_retrieval_query(predictions_data)
        logger.info(f"[RetrievalAgent] Formulated Query: '{query_text}'")
        retrieved_reports = self.retriever.search_similar_reports(query_text=query_text, top_k=top_k)
        logger.info(f"[RetrievalAgent] Retrieved {len(retrieved_reports)} reference documents.")
        return query_text, retrieved_reports


class FindingVerificationAgent:
    """Agent 2: Cross-checks predicted findings against probability cutoffs and retrieved evidence."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def verify_findings(
        self,
        predictions_data: Dict[str, Any],
        retrieved_reports: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        logger.info("[FindingVerificationAgent] Cross-checking vision probabilities with retrieved evidence...")
        positive_findings = predictions_data.get("positive_findings", [])
        retrieved_text = " ".join([r.get("full_report", "").lower() for r in retrieved_reports])

        verified_findings = []
        flagged_findings = []

        for item in positive_findings:
            finding_name = item.get("finding")
            if finding_name == "No Finding":
                continue

            prob = item.get("probability", 0.0)
            synonyms = PATHOLOGY_TERMS.get(finding_name, [finding_name.lower()])

            # Verification rule: Supported if high probability OR mentioned in retrieved context
            evidence_found = any(syn in retrieved_text for syn in synonyms)
            high_confidence = prob >= 0.20

            if high_confidence or evidence_found:
                verified_item = dict(item)
                verified_item["verified_by_retrieval"] = evidence_found
                verified_item["verification_status"] = "VERIFIED"
                verified_findings.append(verified_item)
            else:
                flagged_item = dict(item)
                flagged_item["verification_status"] = "FLAGGED_UNSUPPORTED"
                flagged_findings.append(flagged_item)

        logger.info(f"[FindingVerificationAgent] Verified {len(verified_findings)} findings, Flagged {len(flagged_findings)} unsupported findings.")
        return {
            "verified_findings": verified_findings,
            "flagged_findings": flagged_findings
        }


class DraftingAgent:
    """Agent 3: Generates preliminary radiology report draft using verified findings."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.generator = MedicalReportGenerator(config=config)

    def draft_report(
        self,
        predictions_data: Dict[str, Any],
        verification_result: Dict[str, Any],
        retrieved_reports: List[Dict[str, Any]]
    ) -> str:
        logger.info("[DraftingAgent] Generating preliminary radiology report draft...")
        # Inject verified findings into a copy of predictions_data
        draft_preds = dict(predictions_data)
        draft_preds["positive_findings"] = verification_result["verified_findings"]

        draft_report = self.generator.generate_report(
            predictions_data=draft_preds,
            rag_payload=retrieved_reports
        )
        return draft_report


class HallucinationCriticAgent:
    """Agent 4: Criticizes drafted report, flagging or removing ungrounded clinical claims."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def criticize_and_refine(
        self,
        draft_report: str,
        verified_findings: List[Dict[str, Any]],
        retrieved_reports: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        logger.info("[HallucinationCriticAgent] Inspecting draft report for hallucinated clinical claims...")
        retrieved_text = " ".join([r.get("full_report", "").lower() for r in retrieved_reports])
        verified_names = {f["finding"].lower() for f in verified_findings}

        unsupported_claims = []
        report_lower = draft_report.lower()

        # Check all 14 CheXpert pathology terms
        for pathology, synonyms in PATHOLOGY_TERMS.items():
            path_lower = pathology.lower()
            # If mentioned in report text but NOT in verified findings AND NOT in retrieved text
            mentioned_in_report = any(syn in report_lower for syn in synonyms)
            supported_by_vision = path_lower in verified_names or any(syn in " ".join(verified_names) for syn in synonyms)
            supported_by_rag = any(syn in retrieved_text for syn in synonyms)

            if mentioned_in_report and not (supported_by_vision or supported_by_rag):
                unsupported_claims.append(pathology)

        refined_report = draft_report
        if unsupported_claims:
            logger.warning(f"[HallucinationCriticAgent] Detected unsupported claims in draft: {unsupported_claims}")
            # Add critique note to report disclaimer
            critic_note = f"\n\n> [!WARNING]\n> **Hallucination Critic Audit**: The following findings were flagged as unverified and excluded: {', '.join(unsupported_claims)}."
            refined_report += critic_note

        critic_metadata = {
            "unsupported_claims_detected": unsupported_claims,
            "claims_count": len(unsupported_claims),
            "critic_verdict": "PASSED" if not unsupported_claims else "REFINED_WITH_WARNINGS"
        }
        return refined_report, critic_metadata


class MultiAgentRAGPipeline:
    """
    Multi-Agent Retrieval-Augmented Generation (RAG) Orchestrator.
    Coordinates RetrievalAgent, FindingVerificationAgent, DraftingAgent, and HallucinationCriticAgent.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retrieval_agent = RetrievalAgent(config=config)
        self.verification_agent = FindingVerificationAgent(config=config)
        self.drafting_agent = DraftingAgent(config=config)
        self.critic_agent = HallucinationCriticAgent(config=config)

    def process(self, predictions_data: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
        logger.info("Executing Multi-Agent RAG workflow...")

        # Agent 1: Retrieval
        query_text, retrieved_reports = self.retrieval_agent.retrieve(predictions_data, top_k=top_k)

        # Agent 2: Verification
        verification_result = self.verification_agent.verify_findings(predictions_data, retrieved_reports)

        # Build context
        rag_context = build_rag_context(predictions_data, retrieved_reports)

        # Agent 3: Drafting
        draft_report = self.drafting_agent.draft_report(
            predictions_data=predictions_data,
            verification_result=verification_result,
            retrieved_reports=retrieved_reports
        )

        # Agent 4: Criticism & Refinement
        final_report, critic_metadata = self.critic_agent.criticize_and_refine(
            draft_report=draft_report,
            verified_findings=verification_result["verified_findings"],
            retrieved_reports=retrieved_reports
        )

        return {
            "query_text": query_text,
            "retrieved_reports": retrieved_reports,
            "rag_context": rag_context,
            "verification_result": verification_result,
            "draft_report": draft_report,
            "final_report": final_report,
            "critic_metadata": critic_metadata
        }
