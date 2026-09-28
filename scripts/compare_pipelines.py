"""
Three-Way Pipeline Comparison Experiment Script.
Evaluates:
(a) End-to-End SLM-Only Baseline (no classifier, no RAG)
(b) Single-Agent RAG Pipeline
(c) Multi-Agent RAG Pipeline
on the held-out IU X-Ray test split (N=575).
Measures NLG metrics (BLEU-1..4, ROUGE-L, METEOR, CIDEr), Unsupported Claim Rate (UCR), and Latency (ms/report).
Saves outputs to results/pipeline_comparison.json and results/pipeline_comparison.csv.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config_parser import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.pipeline import RadiologyAIPipeline
from src.data.iu_xray_dataset import IUXRayDatasetLoader
from src.verification.hallucination_metric import calculate_batch_ucr
from evaluate import perform_patient_level_split, compute_cider

logger = setup_logger("compare_pipelines")
smooth = SmoothingFunction().method1
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def run_pipeline_comparison(config_path: str = "configs/config.yaml", max_samples: int = 50):
    config = load_config(config_path)
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    iu_path = config.get("data", {}).get("iu_xray_path", "datasets/iu_xray")
    raw_iu = IUXRayDatasetLoader(root_dir=iu_path, csv_filename="indiana_reports.csv", config=config, is_train=False)

    if not raw_iu or len(raw_iu.samples) == 0:
        logger.error("IU X-Ray dataset not found.")
        return

    _, _, test_indices, _ = perform_patient_level_split(raw_iu, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed)
    test_indices = test_indices[:max_samples]

    # Map image paths
    img_dir = os.path.join(iu_path, "IU X-Ray  Open-I", "images", "images_normalized")
    eval_samples = []

    for idx in test_indices:
        sample = raw_iu.samples[idx]
        uid = str(sample.get("report_id", ""))
        ref_text = sample.get("report", "").strip()

        img_path = sample.get("image_path", "")
        if not os.path.exists(img_path) and os.path.exists(img_dir):
            for fname in os.listdir(img_dir):
                if fname.startswith(f"{uid}_"):
                    img_path = os.path.join(img_dir, fname)
                    break

        if os.path.exists(img_path) and ref_text:
            eval_samples.append({
                "uid": uid,
                "image_path": img_path,
                "reference_report": ref_text
            })

    logger.info(f"Evaluating {len(eval_samples)} test samples across 3 pipeline variants...")

    pipeline_single = RadiologyAIPipeline(config_path=config_path, rag_mode="single_agent")
    pipeline_multi = RadiologyAIPipeline(config_path=config_path, rag_mode="multi_agent")

    variants = ["SLM-Only Baseline", "Single-Agent RAG", "Multi-Agent RAG"]
    variant_results = {}

    for var in variants:
        logger.info(f"--- Running Variant: {var} ---")
        ref_texts = []
        gen_texts = []
        latencies = []
        pos_findings_batch = []
        retrieved_reports_batch = []

        for sample in eval_samples:
            img_path = sample["image_path"]
            ref_text = sample["reference_report"]

            t0 = time.time()
            if var == "SLM-Only Baseline":
                # Run classification only without RAG retrieval context
                res = pipeline_single.analyze_xray(img_path, top_k=0)
                # Baseline generic synthesis without RAG context
                gen_text = res["report"]
                pf = res.get("positive_findings", [])
                rr = []
            elif var == "Single-Agent RAG":
                res = pipeline_single.analyze_xray(img_path, top_k=3)
                gen_text = res["report"]
                pf = res.get("positive_findings", [])
                rr = res.get("retrieved_reports", [])
            else: # Multi-Agent RAG
                res = pipeline_multi.analyze_xray(img_path, top_k=3)
                gen_text = res["report"]
                pf = res.get("positive_findings", [])
                rr = res.get("retrieved_reports", [])

            elapsed_ms = (time.time() - t0) * 1000.0
            latencies.append(elapsed_ms)
            ref_texts.append(ref_text)
            gen_texts.append(gen_text)
            pos_findings_batch.append(pf)
            retrieved_reports_batch.append(rr)

        # Compute NLG Metrics
        b1_list, b2_list, b3_list, b4_list = [], [], [], []
        rouge_list, meteor_list = [], []

        for r, g in zip(ref_texts, gen_texts):
            rt = r.lower().split()
            gt = g.lower().split()
            b1_list.append(sentence_bleu([rt], gt, weights=(1,0,0,0), smoothing_function=smooth))
            b2_list.append(sentence_bleu([rt], gt, weights=(0.5,0.5,0,0), smoothing_function=smooth))
            b3_list.append(sentence_bleu([rt], gt, weights=(0.33,0.33,0.33,0), smoothing_function=smooth))
            b4_list.append(sentence_bleu([rt], gt, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth))
            rouge_list.append(scorer.score(r, g)['rougeL'].fmeasure)
            meteor_list.append(meteor_score([rt], gt))

        cider_score = compute_cider(ref_texts, gen_texts)
        ucr_res = calculate_batch_ucr(gen_texts, pos_findings_batch, retrieved_reports_batch)

        variant_results[var] = {
            "variant": var,
            "sample_count": len(eval_samples),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "bleu_1": round(float(np.mean(b1_list)), 4),
            "bleu_2": round(float(np.mean(b2_list)), 4),
            "bleu_3": round(float(np.mean(b3_list)), 4),
            "bleu_4": round(float(np.mean(b4_list)), 4),
            "rouge_l": round(float(np.mean(rouge_list)), 4),
            "meteor": round(float(np.mean(meteor_list)), 4),
            "cider": round(float(cider_score), 4),
            "unsupported_claim_rate": ucr_res["mean_ucr"],
            "total_unsupported_entities": ucr_res["total_unsupported_entities"]
        }

    results_dir = config.get("paths", {}).get("results_dir", "results")
    json_path = os.path.join(results_dir, "pipeline_comparison.json")
    csv_path = os.path.join(results_dir, "pipeline_comparison.csv")

    with open(json_path, "w") as f:
        json.dump(variant_results, f, indent=4)

    df = pd.DataFrame(list(variant_results.values()))
    df.to_csv(csv_path, index=False)

    logger.info(f"Pipeline comparison complete! Saved to {json_path} and {csv_path}")
    print("\n" + "="*80)
    print("                THREE-WAY PIPELINE COMPARISON RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    run_pipeline_comparison()
