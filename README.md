# A Framework for AI-Based Preliminary Radiology Report Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-VectorStore-green.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

An end-to-end medical AI framework combining multi-label deep learning image classification with Single-Agent Retrieval-Augmented Generation (RAG) to generate grounded preliminary radiology reports from chest X-ray images.

---

## 🩻 System Pipeline Architecture

```
INPUT CHEST X-RAY IMAGE
        ↓
IMAGE PREPROCESSING (CLAHE, Resizing 224x224, Normalization)
        ↓
MULTI-LABEL DISEASE CLASSIFIER (DenseNet121 / Sigmoid Probabilities)
        ↓
FINDINGS + PROBABILITY SCORES (Cardiomegaly 82%, Effusion 71%, Edema 64%)
        ↓
SEMANTIC RETRIEVAL QUERY FORMATION
        ↓
FAISS VECTOR DATABASE SEARCH (IU X-Ray Report Corpus)
        ↓
TOP-K RELEVANT RADIOLOGY REPORTS
        ↓
SINGLE-AGENT RAG CONTEXT ASSEMBLY
        ↓
MEDICAL LANGUAGE MODEL (HuggingFace / Ollama / Grounded Generator Engine)
        ↓
PRELIMINARY RADIOLOGY REPORT (Findings Table, Observations, Impression, Disclaimer)
```

---

## 📁 Dataset Folder Setup Instructions

Please place your medical datasets into the `datasets/` folder structure prior to training:

```
datasets/
├── chexpert/
│   ├── train.csv
│   ├── valid.csv
│   ├── train/
│   └── valid/
├── iu_xray/
│   ├── indiana_reports.csv
│   └── images/
└── README.md
```

- **CheXpert**: Used for training and evaluating the 14 multi-label observation classifier.
- **IU X-Ray**: Used as the semantic retrieval knowledge base corpus for report retrieval.

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Clone or navigate to the project directory
cd "our-project"

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Unit & Integration Tests

```bash
pytest tests/
```

### 3. Training & Evaluation Commands

```bash
# Train multi-label classifier on CheXpert dataset
python run.py train-classifier --dataset chexpert

# Index IU X-Ray radiology reports into FAISS vector database
python run.py build-retrieval

# Evaluate multi-label performance (AUROC, AUPRC, F1, Precision, Recall)
python run.py evaluate

# Predict findings for a single chest X-ray
python run.py predict --image datasets/sample_xray.png

# Run complete End-to-End RAG Report Generation
python run.py generate-report --image datasets/sample_xray.png
```

### 4. Interactive Streamlit Web App

```bash
streamlit run app/streamlit_app.py
```

### 5. Generate Academic Project Report

```bash
# Automatically creates project_report/FINAL_PROJECT_REPORT.md with 30 academic sections
python run.py generate-doc
```

---

## ⚠️ Academic & Medical Disclaimer

> **IMPORTANT**: This system is designed for **academic, research, and educational purposes**. All outputs produced by this framework are explicitly designated as **PRELIMINARY / AI-GENERATED** reports and do **NOT** constitute a final clinical diagnosis. All reports must be reviewed by a qualified radiologist before clinical use.
