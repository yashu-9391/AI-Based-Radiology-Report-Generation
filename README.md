# AI-Based Preliminary Radiology Report Generation Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-VectorStore-green.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

An end-to-end medical AI framework combining multi-label deep learning image classification with a Single-Agent Retrieval-Augmented Generation (RAG) architecture, automatic X-Ray image validation, and IU-XRay ground-truth report verification to generate grounded preliminary radiology reports from chest X-ray images.

---

## 🌟 Key Features

1. **Automatic Chest X-Ray Image Validation Engine**:
   - Inspects input image color saturation, RGB channel variance ($|R-G| + |G-B| + |B-R|$), luminance histogram, and contrast profiles.
   - Automatically detects and rejects non-X-ray images (e.g. general color photos, animals, landscapes, non-radiology images).
   - Triggers an interactive warning popup modal (**"⚠️ Invalid Image Selection: Please select only X-Ray images"**) in the Streamlit UI.

2. **Multi-Label Deep Learning Disease Classifier**:
   - Predicts 14 standard radiological observations (Cardiomegaly, Pleural Effusion, Edema, Atelectasis, Pneumonia, Consolidation, Pneumothorax, etc.) using DenseNet121 / ResNet architectures.

3. **Full IU-XRay Corpus Indexing & Vector Store**:
   - Automatically indexes all **3,826 Indiana University Chest X-Ray (IU-XRay)** reports into a FAISS vector database with sentence-transformer embeddings (`all-MiniLM-L6-v2`).

4. **Single-Agent RAG Report Generation**:
   - Formulates semantic search queries from positive deep learning observations, retrieves semantically matched reference reports, and synthesizes structured preliminary radiology reports.

5. **IU-XRay Ground-Truth Report Verification Engine**:
   - Verifies AI predicted findings and generated reports against ground-truth/retrieved IU-XRay reports.
   - Computes semantic similarity percentages (0–100%) and finding-level alignment badges.

---

## 🩻 System Architecture Workflow

```
                  [ INPUT IMAGE ]
                         ↓
             [ STEP 0: X-RAY IMAGE VALIDATOR ]
           (Rejects Non-X-Ray -> Warning Popup)
                         ↓
           [ STEP 1: IMAGE PREPROCESSING & CLAHE ]
                         ↓
         [ STEP 2: MULTI-LABEL DL CLASSIFIER ]
        (DenseNet121 Sigmoid Probabilities 14 Classes)
                         ↓
           [ STEP 3: SEMANTIC RAG RETRIEVAL ]
         (FAISS Vector DB - 3,826 IU-XRay Reports)
                         ↓
        [ STEP 4: PRELIMINARY REPORT GENERATION ]
         (Structured Table, Observations, Impression)
                         ↓
       [ STEP 5: IU-XRAY GROUND-TRUTH VERIFICATION ]
    (Cosine Similarity Score %, Verified Finding Alignment)
```

---

## 📁 Dataset Directory Setup

Place your medical datasets in the `datasets/` directory:

```
datasets/
├── chexpert/
│   ├── train.csv
│   ├── valid.csv
│   └── train/
├── iu_xray/
│   └── IU X-Ray  Open-I/
│       ├── indiana_reports.csv       (3,826 reports)
│       ├── indiana_projections.csv   (7,466 image projections)
│       └── images/images_normalized/ (7,470 normalized chest X-rays)
```

---

## 🚀 Quick Start & Installation Guide

### 1. Environment Setup & Dependencies

```bash
# Navigate to project folder
cd "our-project"

# Install python dependencies
pip install -r requirements.txt
```

### 2. Interactive Web Application (Streamlit UI)

To launch the web interface:

```bash
python -m streamlit run app/streamlit_app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

- **Non-X-Ray Validation**: Uploading a color photo or non-X-ray image triggers an alert popup (**"Please select only X-Ray images"**).
- **Chest X-Ray Analysis & Verification**: Uploading a valid chest X-ray image generates multi-label disease probabilities, retrieves IU-XRay reference reports, displays ground-truth verification match scores (e.g. `84% Similarity`, `Verified Findings`), and produces a downloadable report.

### 3. Command Line Interface (CLI)

```bash
# 1. Run validation and verification on a Chest X-Ray image
python run.py verify-iu-xray --image "datasets/iu_xray/IU X-Ray  Open-I/images/images_normalized/1_IM-0001-3001.dcm.png"

# 2. Predict multi-label findings for a single X-ray
python run.py predict --image "datasets/iu_xray/IU X-Ray  Open-I/images/images_normalized/1_IM-0001-3001.dcm.png"

# 3. Index IU-XRay reports into FAISS vector database
python run.py build-retrieval

# 4. Train multi-label classifier
python run.py train-classifier --dataset chexpert

# 5. Evaluate classifier performance metrics (AUROC, F1, Precision, Recall)
python run.py evaluate

# 6. Generate 30-section academic project documentation
python run.py generate-doc
```

### 4. Running Unit Tests

Execute the automated pytest test suite (10 unit & integration tests):

```bash
python -m pytest tests/
```

---

## 📊 Evaluation & Verification Metrics

| Metric Component | Method / Source | Target / Standard |
|---|---|---|
| **Image Validation** | Multi-tier saturation & channel variance | Enforces Chest X-Ray selection only |
| **Multi-Label Classification** | DenseNet121 + Weighted BCE | 14 CheXpert/IU-XRay observations |
| **Semantic Retrieval** | FAISS Vector Store + SentenceTransformers | Top-K similarity ranking |
| **Ground-Truth Verification** | Cosine Similarity + Finding Alignment | Semantic match % & confirmed finding badges |

---

## ⚠️ Academic & Medical Disclaimer

> **IMPORTANT**: This system is designed strictly for **academic, research, and educational purposes**. All reports produced by this framework are explicitly designated as **PRELIMINARY / AI-GENERATED** reports and do **NOT** constitute a final medical diagnosis. All reports must be reviewed by a qualified radiologist before clinical use.
