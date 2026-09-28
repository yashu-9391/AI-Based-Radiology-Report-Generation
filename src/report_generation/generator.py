import os
from typing import Dict, Any, List, Union
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

    def build_prompt(self, rag_payload: Dict[str, Any]) -> str:
        """
        Builds the structured medical report prompt using the RAG payload context.
        """
        rag_context = rag_payload.get("rag_context", "")
        return get_medical_report_prompt(rag_context)

    def generate_report(self, predictions_data: Dict[str, Any], rag_payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> str:
        """
        Generates preliminary radiology report using selected provider or rule-based fallback.
        """
        if isinstance(rag_payload, list):
            retrieved_reports = rag_payload
            rag_context = "\n".join([r.get("full_report", "") for r in retrieved_reports])
        else:
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
            hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
            logger.info(f"Loading HuggingFace SLM generator: {self.model_name}")
            prompt = get_medical_report_prompt(rag_context)
            pipe_kwargs = {"max_new_tokens": self.max_tokens, "temperature": self.temperature}
            if hf_token:
                pipe_kwargs["token"] = hf_token
            pipe = pipeline("text-generation", model=self.model_name, **pipe_kwargs)
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
                logger.warning("OPENAI_API_KEY not found in environment.")
                return ""
            prompt = get_medical_report_prompt(rag_context)
            
            if hasattr(openai, "OpenAI"):
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature
                )
                return response.choices[0].message.content.strip()
            else:
                openai.api_key = api_key
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
        Synthesizes a grounded, patient-friendly preliminary chest X-ray explanation in plain English.
        """
        positive_findings = predictions_data.get("positive_findings", [])
        
        # Filter out "No Finding" from positive disease findings
        disease_findings = [p for p in positive_findings if p.get('finding') != "No Finding"]
        
        # Sort disease findings prioritizing active medical pathologies over support devices
        def finding_priority(item):
            name = item.get('finding', '')
            if name == "Support Devices":
                return (1, -item.get('probability', 0))
            return (0, -item.get('probability', 0))

        disease_findings.sort(key=finding_priority)
        top_disease_findings = disease_findings[:5]

        plain_language_map = {
            "Cardiomegaly": ("Heart Outline", "The outline of the heart appears slightly larger than usual on the scan image. This can sometimes be related to blood pressure, natural variations, or fluid."),
            "Lung Opacity": ("Cloudy Area in Lung", "Part of the lung appears denser or cloudier than standard clear air space. This can happen with a chest cold, mucus, or minor inflammation."),
            "Edema": ("Fluid Balance", "Visual patterns suggesting possible extra fluid in the lung tissues."),
            "Consolidation": ("Filled Area in Lung", "An area where air space looks filled or cloudy, which can occur during a chest infection or cold."),
            "Pneumonia": ("Chest Infection Features", "Image patterns that can sometimes be associated with a lung infection or chest cold."),
            "Atelectasis": ("Partial Expansion", "A small section of the lung appears slightly less expanded, which often happens when resting or shallow breathing."),
            "Pneumothorax": ("Air Space Outside Lung", "Signs suggesting air pockets outside the main lung tissue."),
            "Pleural Effusion": ("Fluid Around Lungs", "Features indicating extra fluid accumulated around the outer edges of the lungs."),
            "Pleural Other": ("Lung Lining Variation", "Slight variation observed in the thin lining surrounding the lungs."),
            "Fracture": ("Bone Shadow", "A shadow along a bone (such as a rib) that may suggest a crack or healing past injury."),
            "Support Devices": ("Visible Medical Devices", "Medical equipment such as monitor wires, leads, or sensors are visible on the scan."),
            "Enlarged Cardiomediastinum": ("Central Chest Shadow", "The central chest area near the heart appears slightly wider on the image."),
            "Lung Lesion": ("Localized Spot", "A small spot or shadow was observed in the lung area that requires routine review.")
        }

        # Section 1: What the AI Noticed
        if top_disease_findings:
            finding_labels = [f"**{p['finding']}**" for p in top_disease_findings]
            noticed_str = (
                f"The AI analyzed your chest X-ray and noticed a few visual patterns that differ from a standard clear image. "
                f"The main observations include: {', '.join(finding_labels)}."
            )
        else:
            noticed_str = "The AI analyzed your chest X-ray and did not detect significant abnormal findings. The lung fields and heart outline appear clear."

        # Section 2: What These Findings Mean in Simple Terms
        meaning_lines = []
        if top_disease_findings:
            for p in top_disease_findings:
                fname = p['finding']
                friendly_title, friendly_explanation = plain_language_map.get(
                    fname,
                    (fname, "A visual feature detected on the image that differs slightly from normal.")
                )
                meaning_lines.append(f"- **{friendly_title} ({fname}):** {friendly_explanation}")
            meaning_str = "\n".join(meaning_lines)
        else:
            meaning_str = "No specific abnormal features were highlighted by the AI model on this scan."

        # Section 3: Why This Is NOT a Confirmed Diagnosis
        assoc_str = (
            "An AI detecting a shadow or pattern on an image does **NOT** mean you have a specific disease. "
            "Many common, mild, or temporary conditions (such as a recent cold, past infection, or simple positioning during the X-ray) "
            "can produce similar visual features. A full diagnosis can only be made by a doctor reviewing your symptoms and physical exam."
        )

        # Section 4: What You Should Do Next
        next_str = (
            "1. **Share with your Doctor:** Show this preliminary note and your original X-ray image to your physician or radiologist.\n"
            "2. **Discuss your Symptoms:** Tell your healthcare provider if you have been experiencing cough, fever, shortness of breath, or chest pain.\n"
            "3. **Follow Professional Advice:** Your doctor will evaluate the X-ray in combination with how you feel to determine if any simple follow-up is needed."
        )

        # Section 5: When to Seek Prompt Care
        urgent_str = "If you experience severe shortness of breath, sudden sharp chest pain, high fever, or confusion, please seek immediate medical attention."

        # Section 6: Overall Impression
        if top_disease_findings:
            titles = [p['finding'] for p in top_disease_findings]
            impression_str = (
                f"The AI highlighted preliminary visual features ({', '.join(titles)}). "
                f"These observations are non-diagnostic and should be reviewed by a qualified doctor alongside your clinical symptoms."
            )
        else:
            impression_str = "The chest X-ray image analysis showed no significant acute cardiopulmonary abnormalities."

        # Section 7: Important Disclaimer
        disclaimer_str = (
            "This is an AI-generated preliminary explanation created for informational and educational purposes only. "
            "It is NOT a confirmed medical diagnosis. Always consult a qualified radiologist or physician for medical interpretation."
        )

        full_report = (
            f"# PRELIMINARY CHEST X-RAY EXPLANATION\n\n"
            f"## 1. What the AI Observed\n\n{noticed_str}\n\n"
            f"## 2. What These Findings Mean in Simple Language\n\n{meaning_str}\n\n"
            f"## 3. Why This Is NOT a Confirmed Diagnosis\n\n{assoc_str}\n\n"
            f"## 4. Recommended Next Steps\n\n{next_str}\n\n"
            f"## 5. When to Seek Urgent Care\n\n{urgent_str}\n\n"
            f"## 6. Summary\n\n{impression_str}\n\n"
            f"## 7. Important Disclaimer\n\n{disclaimer_str}"
        )
        return full_report



