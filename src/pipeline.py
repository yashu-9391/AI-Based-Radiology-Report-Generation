import os
from typing import Dict, Any, Union, Optional
from PIL import Image
import numpy as np

from src.utils.config_parser import load_config
from src.utils.logger import setup_logger
from src.classification.predict import predict_image
from src.rag.rag_pipeline import SingleAgentRAGPipeline
from src.rag.multi_agent_pipeline import MultiAgentRAGPipeline
from src.report_generation.generator import MedicalReportGenerator

from src.preprocessing.image_validator import validate_chest_xray
from src.verification.iu_xray_verifier import IUXRayReportVerifier

logger = setup_logger(__name__)

class RadiologyAIPipeline:
    """
    End-to-End Radiology AI Pipeline for Preliminary Report Generation.
    Integrates image validation, multi-label deep learning classification,
    semantic vector retrieval, single-agent / multi-agent RAG, grounded medical report generation,
    and IU-XRay corpus verification.
    """
    def __init__(self, config_path: str = "configs/config.yaml", rag_mode: Optional[str] = None):
        self.config = load_config(config_path)
        selected_rag_mode = (rag_mode or self.config.get("rag", {}).get("mode", "single_agent")).lower()
        if selected_rag_mode == "multi_agent":
            self.rag_pipeline = MultiAgentRAGPipeline(config=self.config)
            logger.info("Initialized Radiology AI Pipeline with MULTI-AGENT RAG mode.")
        else:
            self.rag_pipeline = SingleAgentRAGPipeline(config=self.config)
            logger.info("Initialized Radiology AI Pipeline with SINGLE-AGENT RAG mode.")

        self.generator = MedicalReportGenerator(config=self.config)
        self.verifier = IUXRayReportVerifier(config=self.config)

    def analyze_xray(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        model_path: str = "models/best_model.pth",
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main execution workflow:
        Validation -> Preprocessing -> Multi-label Classifier -> RAG -> Report -> Verification.
        """
        logger.info("Step 0: Validating Chest X-Ray Image...")
        is_valid, error_msg, validation_metrics = validate_chest_xray(image_input=image_input)
        if not is_valid:
            logger.error(f"Image validation failed: {error_msg}")
            raise ValueError(error_msg)

        logger.info("Step 1: Running Multi-Label Disease Classifier...")
        predictions_data = predict_image(
            image_input=image_input,
            config=self.config,
            model_path=model_path
        )

        logger.info("Step 2 & 3: Executing RAG and Semantic Retrieval...")
        k = top_k or self.config.get("retrieval", {}).get("top_k", 3)
        rag_payload = self.rag_pipeline.process(predictions_data=predictions_data, top_k=k)

        logger.info("Step 4: Generating Grounded Preliminary Radiology Report...")
        if "final_report" in rag_payload:
            generated_report = rag_payload["final_report"]
        else:
            generated_report = self.generator.generate_report(
                predictions_data=predictions_data,
                rag_payload=rag_payload
            )

        logger.info("Step 5: Executing Verification with IU-XRay Dataset...")
        verification_result = self.verifier.verify_report(
            generated_report=generated_report,
            predictions_data=predictions_data,
            retrieved_reports=rag_payload["retrieved_reports"]
        )

        result = {
            "image": predictions_data["image"],
            "validation": validation_metrics,
            "threshold": predictions_data["threshold"],
            "predictions": predictions_data["predictions"],
            "positive_findings": predictions_data["positive_findings"],
            "query_text": rag_payload["query_text"],
            "retrieved_reports": rag_payload["retrieved_reports"],
            "rag_context": rag_payload["rag_context"],
            "prompt": self.generator.build_prompt(rag_payload),
            "report": generated_report,
            "verification": verification_result
        }

        logger.info("End-to-End Analysis completed successfully.")
        return result

def analyze_xray(
    image_input: Union[str, Image.Image, np.ndarray],
    config_path: str = "configs/config.yaml",
    model_path: str = "models/best_model.pth"
) -> Dict[str, Any]:
    """
    Convenience functional interface for analyze_xray.
    """
    pipeline = RadiologyAIPipeline(config_path=config_path)
    return pipeline.analyze_xray(image_input=image_input, model_path=model_path)
