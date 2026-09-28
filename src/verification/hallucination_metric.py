"""
Quantitative Hallucination Metric: Unsupported Claim Rate (UCR).
Evaluates natural language radiology reports for ungrounded or hallucinated clinical claims.
Compares mentioned clinical entities in generated reports against:
(a) Calibrated vision classifier positive findings
(b) Retrieved reference radiology reports context
"""

import re
from typing import Dict, Any, List, Union

# Fixed clinical entity vocabulary covering 14 CheXpert observations and clinical synonyms
CLINICAL_ENTITY_VOCAB = {
    "Enlarged Cardiomediastinum": [r"\benlarged cardiomediastinum\b", r"\bmediastin\w*\b", r"\bcentral chest shadow\b"],
    "Cardiomegaly": [r"\bcardiomegaly\b", r"\benlarged heart\b", r"\bcardiac outline\b", r"\bcardiac silhouette\b"],
    "Lung Opacity": [r"\blung opacity\b", r"\bopacit\w*\b", r"\bcloudy area\b", r"\binfiltrate\w*\b", r"\bshadow\b"],
    "Lung Lesion": [r"\blung lesion\b", r"\blesion\b", r"\bnodule\b", r"\bmass\b", r"\blocalized spot\b"],
    "Edema": [r"\bedema\b", r"\bfluid balance\b", r"\bcongestion\b", r"\bvascular overload\b"],
    "Consolidation": [r"\bconsolidation\b", r"\bfilled area in lung\b", r"\bairspace opacity\b"],
    "Pneumonia": [r"\bpneumonia\b", r"\bchest infection\b", r"\blung infection\b", r"\binfectious opacity\b"],
    "Atelectasis": [r"\batelectasis\b", r"\bpartial expansion\b", r"\bcollapsed lung\b", r"\bvolume loss\b"],
    "Pneumothorax": [r"\bpneumothorax\b", r"\bair space outside lung\b", r"\bair pocket\b"],
    "Pleural Effusion": [r"\bpleural effusion\b", r"\bfluid around lung\b", r"\beffusion\b"],
    "Pleural Other": [r"\bpleural other\b", r"\bpleural lining\b", r"\bpleural thickening\b"],
    "Fracture": [r"\bfracture\b", r"\bbone shadow\b", r"\bbroken rib\b"],
    "Support Devices": [r"\bsupport device\b", r"\bmedical device\b", r"\bcatheter\b", r"\bwire\b", r"\blead\b"],
}

def extract_mentioned_entities(text: str) -> List[str]:
    """
    Extracts clinical pathology entity names present in report text.
    """
    text_lower = text.lower()
    mentioned = []
    for entity_name, patterns in CLINICAL_ENTITY_VOCAB.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                mentioned.append(entity_name)
                break
    return mentioned

def calculate_report_ucr(
    report_text: str,
    positive_findings: List[Dict[str, Any]],
    retrieved_reports: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculates Unsupported Claim Rate (UCR) for a single report.
    UCR = (Unsupported Mentioned Entities) / (Total Mentioned Entities)
    """
    mentioned_entities = extract_mentioned_entities(report_text)
    if not mentioned_entities:
        return {
            "unsupported_claim_rate": 0.0,
            "total_entities_mentioned": 0,
            "unsupported_count": 0,
            "unsupported_entities": [],
            "supported_entities": []
        }

    # Build supported set from positive vision predictions
    supported_set = set()
    for pf in positive_findings:
        finding = pf.get("finding")
        if finding and finding != "No Finding":
            supported_set.add(finding)

    # Build supported set from retrieved context
    retrieved_text = " ".join([r.get("full_report", "") for r in retrieved_reports])
    retrieved_entities = extract_mentioned_entities(retrieved_text)
    supported_set.update(retrieved_entities)

    unsupported_entities = []
    supported_entities = []

    for entity in mentioned_entities:
        if entity in supported_set:
            supported_entities.append(entity)
        else:
            unsupported_entities.append(entity)

    ucr = len(unsupported_entities) / len(mentioned_entities)

    return {
        "unsupported_claim_rate": round(ucr, 4),
        "total_entities_mentioned": len(mentioned_entities),
        "unsupported_count": len(unsupported_entities),
        "unsupported_entities": unsupported_entities,
        "supported_entities": supported_entities
    }

def calculate_batch_ucr(
    reports: List[str],
    positive_findings_list: List[List[Dict[str, Any]]],
    retrieved_reports_list: List[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Calculates average UCR over a batch/dataset of reports.
    """
    scores = []
    total_unsupported = 0
    total_entities = 0

    for rep, pf, rr in zip(reports, positive_findings_list, retrieved_reports_list):
        res = calculate_report_ucr(rep, pf, rr)
        scores.append(res["unsupported_claim_rate"])
        total_unsupported += res["unsupported_count"]
        total_entities += res["total_entities_mentioned"]

    mean_ucr = sum(scores) / len(scores) if scores else 0.0
    macro_ucr = total_unsupported / total_entities if total_entities > 0 else 0.0

    return {
        "mean_ucr": round(mean_ucr, 4),
        "macro_ucr": round(macro_ucr, 4),
        "total_reports_evaluated": len(reports),
        "total_entities_evaluated": total_entities,
        "total_unsupported_entities": total_unsupported
    }
