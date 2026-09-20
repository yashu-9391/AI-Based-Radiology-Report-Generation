import os
from typing import Dict, Any, List
from src.report_generation.prompt_template import get_medical_report_prompt
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class MedicalReportGenerator:
    """
    Medical Language Model report generator.
    Supports HuggingFace SLMs, Ollama, OpenAI API, and a robust grounded Rule-Based engine fallback.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        rg_cfg = config.get("report_generation", {})
        self.provider = rg_cfg.get("llm_provider", "rule_based").lower()
        self.model_name = rg_cfg.get("model_name", "BioMistral/BioMistral-7B")
        self.max_tokens = rg_cfg.get("max_new_tokens", 300)
        self.temperature = rg_cfg.get("temperature", 0.3)

    def generate_report(self, predictions_data: Dict[str, Any], rag_payload: Dict[str, Any]) -> str:
        """
        Generates preliminary radiology report using selected provider or rule-based fallback.
        """
        rag_context = rag_payload.get("rag_context", "")
        retrieved_reports = rag_payload.get("retrieved_reports", [])

        if self.provider == "huggingface":
            report_text = self._generate_huggingface(rag_context)
            if report_text:
                return report_text
        elif self.provider == "ollama":
            report_text = self._generate_ollama(rag_context)
            if report_text:
                return report_text
        elif self.provider == "openai":
            report_text = self._generate_openai(rag_context)
            if report_text:
                return report_text

        # Default & Fallback: Grounded Rule-Based Medical Report Synthesis Engine
        logger.info("Generating report using Grounded Rule-Based Medical Generator Engine...")
        return self._generate_rule_based(predictions_data, retrieved_reports)

    def _generate_huggingface(self, rag_context: str) -> str:
        try:
            from transformers import pipeline
            logger.info(f"Loading HuggingFace SLM generator: {self.model_name}")
            prompt = get_medical_report_prompt(rag_context)
            pipe = pipeline("text-generation", model=self.model_name, max_new_tokens=self.max_tokens, temperature=self.temperature)
            output = pipe(prompt)[0]["generated_text"]
            return output.replace(prompt, "").strip()
        except Exception as e:
            logger.warning(f"HuggingFace SLM generation failed ({e}). Falling back to rule-based engine.")
            return ""

    def _generate_ollama(self, rag_context: str) -> str:
        try:
            import requests
            prompt = get_medical_report_prompt(rag_context)
            response = requests.post("http://localhost:11434/api/generate", json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }, timeout=10)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"Ollama generation failed ({e}). Falling back to rule-based engine.")
        return ""

    def _generate_openai(self, rag_context: str) -> str:
        try:
            import openai
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return ""
            prompt = get_medical_report_prompt(rag_context)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI API generation failed ({e}). Falling back to rule-based engine.")
            return ""

    def _generate_rule_based(self, predictions_data: Dict[str, Any], retrieved_reports: List[Dict[str, Any]]) -> str:
        """
        Synthesizes a grounded preliminary radiology report directly from predictions and retrieved reports.
        """
        positive_findings = predictions_data.get("positive_findings", [])
        all_predictions = predictions_data.get("predictions", [])

        # Section 1: Table of predicted findings
        table_lines = ["| Finding | Probability | Status |", "|---|---|---|"]
        for p in all_predictions[:6]:  # Top 6 findings
            table_lines.append(f"| {p['finding']} | {p['percentage']} | {p['prediction']} |")
        table_str = "\n".join(table_lines)

        # Section 2: Observations / Findings text
        obs_sentences = []
        if positive_findings:
            pos_names = [f"{p['finding']} ({p['percentage']})" for p in positive_findings]
            obs_sentences.append(f"Deep learning image analysis reveals radiographic evidence of {', '.join(pos_names)}.")
        else:
            obs_sentences.append("The lungs demonstrate clear lung fields without focal consolidation, pneumothorax, or overt pleural effusion.")

        # Incorporate retrieved reference report observations
        if retrieved_reports:
            top_ref = retrieved_reports[0]
            ref_findings = top_ref.get("findings", "").strip()
            if ref_findings and len(ref_findings) > 10:
                obs_sentences.append(f"Reference contextual findings (IU X-Ray corpus): {ref_findings}")

        obs_sentences.append("The cardiac silhouette and mediastinal contours are evaluated.")
        observations_str = " ".join(obs_sentences)

        # Section 3: Impression
        if positive_findings:
            pos_titles = [p['finding'] for p in positive_findings]
            impression_str = f"Radiographic findings suspicious for {', '.join(pos_titles)}. Clinical correlation and radiologist verification recommended."
        else:
            impression_str = "No acute cardiopulmonary disease identified by multi-label image analysis."

        # Section 4: AI Disclaimer
        disclaimer_str = (
            "This is an AI-generated preliminary report intended strictly for research and educational purposes. "
            "It does NOT constitute a final medical diagnosis and must be reviewed by a qualified radiologist."
        )

        full_report = (
            f"PRELIMINARY RADIOLOGY REPORT\n\n"
            f"1. PREDICTED FINDINGS TABLE\n{table_str}\n\n"
            f"2. FINDINGS / OBSERVATIONS\n{observations_str}\n\n"
            f"3. IMPRESSION\n{impression_str}\n\n"
            f"4. AI DISCLAIMER\n{disclaimer_str}"
        )
        return full_report
