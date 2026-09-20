from typing import Dict, Any, List

def build_rag_context(predictions_data: Dict[str, Any], retrieved_reports: List[Dict[str, Any]]) -> str:
    """
    Assembles structured context combining deep learning predictions, probabilities, and retrieved reference reports.
    """
    predictions = predictions_data.get("predictions", [])
    positive_findings = predictions_data.get("positive_findings", [])

    # Format 1: Multi-label classification summary
    pred_summary_lines = []
    for pred in predictions:
        pred_summary_lines.append(
            f"- {pred['finding']}: {pred['percentage']} (Probability: {pred['probability']:.2f}) -> {pred['prediction']}"
        )
    predictions_context_str = "\n".join(pred_summary_lines)

    # Format 2: Top-K retrieved reports context
    retrieved_summary_lines = []
    if retrieved_reports:
        for idx, rep in enumerate(retrieved_reports, 1):
            score = rep.get("similarity_score", 0.0)
            score_str = f"{int(round(score * 100))}%" if score <= 1.0 else f"{score:.2f}"
            findings = rep.get("findings", "").strip()
            impression = rep.get("impression", "").strip()
            retrieved_summary_lines.append(
                f"[Reference Report {idx} (Similarity: {score_str})]\nFindings: {findings}\nImpression: {impression}\n"
            )
        retrieved_context_str = "\n".join(retrieved_summary_lines)
    else:
        retrieved_context_str = "No reference reports retrieved."

    # Format 3: Final grounded prompt structure
    full_context = (
        f"=== PREDICTED MULTI-LABEL RADIOGRAPHIC FINDINGS ===\n"
        f"{predictions_context_str}\n\n"
        f"=== RETRIEVED REFERENCE RADIOLOGY REPORTS ===\n"
        f"{retrieved_context_str}\n\n"
        f"=== INSTRUCTIONS ===\n"
        f"Generate a concise preliminary radiology report based STRICTLY on the predicted findings and retrieved reference context.\n"
        f"Do NOT invent unmentioned conditions, numerical measurements, or clinical histories.\n"
        f"Structure the report into: 1. Predicted Findings Table, 2. Findings / Observations, 3. Impression, and 4. Disclaimer."
    )

    return full_context
