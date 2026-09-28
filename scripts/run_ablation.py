"""
Quantitative Ablation Study Script.
Measures NLG metrics, Unsupported Claim Rate (UCR), and latency deltas across:
(i) Without RAG retrieval context (K=0) vs Proposed (K=3)
(ii) Without Youden's J calibration (fixed 0.5 threshold) vs Proposed Youden J cutoffs
(iii) Top-K retrieval sweep over K = {1, 3, 5, 10}
Saves results to results/ablation_results.json and results/ablation_results.csv.
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

logger = setup_logger("ablation_study")
smooth = SmoothingFunction().method1
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def run_ablation_study(config_path: str = "configs/config.yaml", max_samples: int = 40):
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

    logger.info(f"Running Ablation Study on {len(eval_samples)} test samples...")

    # Configurations to evaluate:
    # 1. Proposed Full Pipeline (Youden J + K=3)
    # 2. No RAG Context (K=0)
    # 3. Fixed 0.5 Threshold (No Youden J Calibration)
    # 4. Top-K Sweeps: K=1, K=5, K=10

    ablation_experiments = [
        {"name": "Proposed Full Model (Youden J, K=3)", "top_k": 3, "use_youden": True},
        {"name": "Ablation: No RAG Context (K=0)", "top_k": 0, "use_youden": True},
        {"name": "Ablation: Fixed 0.5 Threshold (No Calibration)", "top_k": 3, "use_youden": False},
        {"name": "Sweep: Top-K = 1", "top_k": 1, "use_youden": True},
        {"name": "Sweep: Top-K = 5", "top_k": 5, "use_youden": True},
        {"name": "Sweep: Top-K = 10", "top_k": 10, "use_youden": True},
    ]

    ablation_results = {}

    for exp in ablation_experiments:
        exp_name = exp["name"]
        top_k = exp["top_k"]
        use_youden = exp["use_youden"]

        logger.info(f"--- Running Ablation Configuration: {exp_name} ---")

        # Custom config setup
        exp_config = dict(config)
        exp_config["classification"] = dict(config.get("classification", {}))
        exp_config["classification"]["use_youden_j"] = use_youden
        if not use_youden:
            exp_config["classification"]["threshold"] = 0.50
            exp_config["classification"]["optimal_thresholds"] = {}

        pipeline = RadiologyAIPipeline(config_path=config_path, rag_mode="single_agent")
        pipeline.config = exp_config  # Override runtime config

        ref_texts = []
        gen_texts = []
        latencies = []
        pos_findings_batch = []
        retrieved_reports_batch = []

        for sample in eval_samples:
            img_path = sample["image_path"]
            ref_text = sample["reference_report"]

            t0 = time.time()
            res = pipeline.analyze_xray(img_path, top_k=top_k)
            elapsed_ms = (time.time() - t0) * 1000.0

            latencies.append(elapsed_ms)
            ref_texts.append(ref_text)
            gen_texts.append(res["report"])
            pos_findings_batch.append(res.get("positive_findings", []))
            retrieved_reports_batch.append(res.get("retrieved_reports", []))

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

        ablation_results[exp_name] = {
            "experiment": exp_name,
            "top_k": top_k,
            "use_youden_j": use_youden,
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "bleu_1": round(float(np.mean(b1_list)), 4),
            "bleu_4": round(float(np.mean(b4_list)), 4),
            "rouge_l": round(float(np.mean(rouge_list)), 4),
            "meteor": round(float(np.mean(meteor_list)), 4),
            "cider": round(float(cider_score), 4),
            "unsupported_claim_rate": ucr_res["mean_ucr"]
        }

    results_dir = config.get("paths", {}).get("results_dir", "results")
    json_path = os.path.join(results_dir, "ablation_results.json")
    csv_path = os.path.join(results_dir, "ablation_results.csv")

    with open(json_path, "w") as f:
        json.dump(ablation_results, f, indent=4)

    df = pd.DataFrame(list(ablation_results.values()))
    df.to_csv(csv_path, index=False)

    logger.info(f"Ablation study complete! Saved to {json_path} and {csv_path}")
    print("\n" + "="*80)
    print("                    QUANTITATIVE ABLATION RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    run_ablation_study()
