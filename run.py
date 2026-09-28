import os
import sys

# Ensure project root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
from PIL import Image

from src.utils.config_parser import load_config
from src.utils.logger import setup_logger

from src.classification.train import train_model
from src.classification.predict import predict_image
from src.classification.evaluation import evaluate_classifier
from src.preprocessing.image_validator import validate_chest_xray
from src.retrieval.retriever import ReportRetriever
from src.verification.iu_xray_verifier import IUXRayReportVerifier
from src.pipeline import analyze_xray
from project_report.generate_academic_report import generate_project_report

logger = setup_logger("cli")

def main():
    parser = argparse.ArgumentParser(description="AI-Based Preliminary Radiology Report Generation CLI Framework")
    subparsers = parser.add_subparsers(dest="command", help="Available CLI Commands")

    # Command: train-classifier
    train_parser = subparsers.add_parser("train-classifier", help="Train multi-label chest X-ray classifier")
    train_parser.add_argument("--dataset", type=str, default="chexpert", choices=["chexpert", "iu_xray"], help="Dataset to train on")
    train_parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")

    # Command: build-retrieval
    build_ret_parser = subparsers.add_parser("build-retrieval", help="Index IU X-Ray reports into FAISS vector database")
    build_ret_parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate classifier multi-label performance metrics")
    eval_parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    eval_parser.add_argument("--model", type=str, default="models/best_model.pth", help="Model checkpoint path")
    eval_parser.add_argument("--rag-mode", type=str, choices=["single_agent", "multi_agent"], default=None, help="RAG orchestration mode")

    # Command: predict
    pred_parser = subparsers.add_parser("predict", help="Predict multi-label findings for a single chest X-ray image")
    pred_parser.add_argument("--image", type=str, required=True, help="Path to input X-ray image")
    pred_parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    pred_parser.add_argument("--model", type=str, default="models/best_model.pth", help="Model checkpoint path")

    # Command: generate-report
    gen_parser = subparsers.add_parser("generate-report", help="Run end-to-end RAG pipeline to generate preliminary report")
    gen_parser.add_argument("--image", type=str, required=True, help="Path to input X-ray image")
    gen_parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    gen_parser.add_argument("--model", type=str, default="models/best_model.pth", help="Model checkpoint path")
    gen_parser.add_argument("--rag-mode", type=str, choices=["single_agent", "multi_agent"], default=None, help="RAG mode: single_agent or multi_agent")

    # Command: verify-iu-xray
    ver_parser = subparsers.add_parser("verify-iu-xray", help="Run validation and verify report against IU X-Ray dataset corpus")
    ver_parser.add_argument("--image", type=str, required=True, help="Path to input X-ray image")
    ver_parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    ver_parser.add_argument("--model", type=str, default="models/best_model.pth", help="Model checkpoint path")
    ver_parser.add_argument("--rag-mode", type=str, choices=["single_agent", "multi_agent"], default=None, help="RAG mode: single_agent or multi_agent")

    # Research Comparative Study Commands
    subparsers.add_parser("compare-pipelines", help="Run 3-way pipeline comparison (SLM-Only, Single-Agent RAG, Multi-Agent RAG)")
    subparsers.add_parser("run-ablation", help="Run quantitative ablation study and Top-K retrieval parameter sweep")
    subparsers.add_parser("compare-backbones", help="Run vision backbone comparison (DenseNet121, ResNet50, EfficientNet-B0)")
    subparsers.add_parser("human-eval", help="Analyze human evaluation ratings and compute Cohen's Kappa agreement")

    # Command: generate-doc
    subparsers.add_parser("generate-doc", help="Generate final academic project report")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "train-classifier":
        config = load_config(args.config)
        logger.info(f"Initiating classifier training on dataset '{args.dataset}'...")
        res = train_model(config=config, dataset_type=args.dataset)
        print("\nTraining Result:", res)

    elif args.command == "build-retrieval":
        config = load_config(args.config)
        retriever = ReportRetriever(config=config)
        count = retriever.build_index_from_dataset()
        print(f"\nIndexed {count} reports into FAISS vector database.")

    elif args.command == "evaluate":
        config = load_config(args.config)
        metrics = evaluate_classifier(config=config, model_path=args.model)
        print("\nEvaluation Metrics Summary:\n", metrics)

    elif args.command == "predict":
        config = load_config(args.config)
        is_valid, err_msg, metrics = validate_chest_xray(args.image)
        if not is_valid:
            print(f"\n[VALIDATION REJECTED]: {err_msg}")
            sys.exit(1)
        res = predict_image(image_input=args.image, config=config, model_path=args.model)
        print("\n--- MULTI-LABEL PREDICTIONS ---")
        for p in res["predictions"]:
            print(f"{p['finding']:<28} | {p['percentage']:<5} | Status: {p['prediction']}")

    elif args.command == "compare-pipelines":
        from scripts.compare_pipelines import run_pipeline_comparison
        run_pipeline_comparison()

    elif args.command == "run-ablation":
        from scripts.run_ablation import run_ablation_study
        run_ablation_study()

    elif args.command == "compare-backbones":
        from scripts.train_and_eval_backbones import run_backbone_comparison
        run_backbone_comparison()

    elif args.command == "human-eval":
        from scripts.compute_human_eval_agreement import analyze_human_eval_ratings
        analyze_human_eval_ratings()

    elif args.command == "generate-report" or args.command == "verify-iu-xray":
        try:
            rag_mode = getattr(args, "rag_mode", None)
            res = analyze_xray(image_input=args.image, config_path=args.config, model_path=args.model)
            print("\n" + "="*60)
            print("                AI PRELIMINARY RADIOLOGY REPORT")
            print("="*60)
            print(res["report"])
            print("="*60)

            ver = res.get("verification", {})
            if ver:
                print("\n--- IU X-RAY DATASET VERIFICATION ---")
                print(f"Status:               {ver.get('status')}")
                print(f"Semantic Similarity:  {ver.get('similarity_percentage')}")
                print(f"Matched Findings:     {', '.join(ver.get('matched_findings', [])) or 'None'}")
                print(f"Summary:              {ver.get('verification_summary')}")
                print("="*60)
        except ValueError as ve:
            print(f"\n[ERROR]: {ve}")


    elif args.command == "generate-doc":
        report_path = generate_project_report()
        print(f"\nAcademic report generated at: {report_path}")

if __name__ == "__main__":
    main()
