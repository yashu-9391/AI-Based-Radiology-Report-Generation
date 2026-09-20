from typing import Dict, Any, List

def build_retrieval_query(predictions_data: Dict[str, Any]) -> str:
    """
    Constructs a semantic search query string from multi-label classifier predictions and probabilities.
    """
    positive_findings = predictions_data.get("positive_findings", [])

    if not positive_findings:
        # Check top prediction even if below default threshold
        all_preds = predictions_data.get("predictions", [])
        if all_preds:
            top_finding = all_preds[0]
            if top_finding["probability"] > 0.35 and top_finding["finding"] != "No Finding":
                return f"{top_finding['finding']} chest X-ray observation"

        return "normal clear lungs cardiac silhouette normal no acute cardiopulmonary finding"

    # Build query from high-probability positive findings
    query_parts = []
    for item in positive_findings:
        finding = item["finding"]
        percentage = item["percentage"]
        query_parts.append(f"{finding} {percentage}")

    query_str = " ".join(query_parts) + " chest radiology findings"
    return query_str
