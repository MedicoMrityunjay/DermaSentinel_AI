---
title: DermaSentinel 🛡️
emoji: 🛡️
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# DermaSentinel: Clinical-Grade AI for Melanoma Detection

![DermaSentinel Banner](https://img.shields.io/badge/Status-Gold%20Master%20v3.0-gold?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker)

**DermaSentinel** is a state-of-the-art, multimodal AI system designed to assist dermatologists in the early detection of melanoma. By fusing deep learning with clinical heuristics (ABCD rule), it provides not just a diagnosis, but a comprehensive, explainable, and equitable clinical assessment.

---

## 🚀 Executive Summary

Melanoma is the deadliest form of skin cancer, but it is highly curable if detected early. DermaSentinel bridges the gap between black-box AI and clinical practice by offering:
1.  **Precision Diagnosis**: A dual-gate architecture (Segmentation + Classification) for high-accuracy lesion analysis.
2.  **Explainability**: Real-time quantification of the **ABCD Rule** (Asymmetry, Border, Color) and visual segmentation overlays.
3.  **Equity**: An integrated **Equity Engine** that analyzes skin tone (ITA Score) to ensure fair performance across diverse skin phenotypes.
4.  **Workflow Integration**: Automated **AI Scribe** for clinical note generation and PDF reporting.

---

## 🌟 Key Features

### 🧠 1. Dual-Gate AI Architecture
*   **Gate 1: The Scalpel (Segmentation)**: A U-Net (ResNet34 encoder) that precisely isolates the lesion from healthy skin.
    *   *Metric*: Dice Coefficient > 0.92
*   **Gate 2: The Fusion (Classification)**: An EfficientNet-B3 backbone fused with patient metadata (Age, Sex, Site) for robust malignancy prediction.
    *   *Metric*: AUC > 0.96

### 📝 2. AI Scribe & Consultant
*   **Automated Notes**: Uses **BLIP (Bootstrapping Language-Image Pre-training)** to generate natural language descriptions of the lesion, synthesized with clinical findings.
*   **Interactive Consultant**: A VQA (Visual Question Answering) module allowing clinicians to ask questions like *"Is the border irregular?"* or *"Describe the color pattern."*

### ⚖️ 3. Equity Engine
*   **Skin Tone Analysis**: Calculates the **Individual Typology Angle (ITA)** in CIELab color space to classify skin type (Fitzpatrick I-VI).
*   **Bias Mitigation**: Automatically flags high-melanin samples (Type V/VI) where segmentation contrast might be lower, alerting the clinician to potential uncertainty.

### 🔍 4. Explainable Concepts (ABCD)
*   **Asymmetry**: Geometric analysis of the lesion mask.
*   **Border**: Compactness and irregularity scoring.
*   **Color**: Standard deviation analysis of RGB channels within the lesion.

### 🛡️ 5. Robustness & Safety
*   **Uncertainty Quantification**: Uses **Test-Time Augmentation (TTA)** (5-10 passes) to generate a confidence interval (e.g., 95% ± 2%).
*   **Quality Gate**: Laplacian variance detection rejects blurry images before analysis.
*   **Input Validation**: Strict server-side validation for age (0-120) and file size (<10MB).

---

## 📊 Performance Metrics (Validation)

| Metric | Value | Description |
| :--- | :--- | :--- |
| **AUC-ROC** | **0.965** | Area Under the Receiver Operating Characteristic Curve |
| **Sensitivity** | **94.2%** | Ability to correctly identify melanoma (True Positive Rate) |
| **Specificity** | **88.5%** | Ability to correctly identify benign lesions (True Negative Rate) |
| **Dice Score** | **0.92** | Segmentation overlap accuracy |
| **Inference** | **< 1.5s** | Average end-to-end processing time on GPU |

*> Note: Metrics based on internal validation set (ISIC 2018 / Custom Split).*

---

## 🛠️ System Architecture

The system is built as a microservice-ready containerized application:

*   **Frontend**: HTML5/JS with Glassmorphism UI, 3D interactions, and Chart.js visualizations.
*   **Backend**: FastAPI (Python 3.10) serving REST endpoints.
*   **AI Engine**: PyTorch with Albumentations for preprocessing and TTA.
*   **Database**: SQLite (via SQLAlchemy) for patient history and feedback tracking.
*   **Infrastructure**: Dockerized for deployment on Hugging Face Spaces or local GPU clusters.

---

## 💻 Installation & Usage

### Option 1: Live Demo
Access the live application on Hugging Face Spaces:
[**DermaSentinel Live**](https://huggingface.co/spaces/medicomrityunjay/DermaSentinel)

### Option 2: Local Deployment (Docker)

```bash
# 1. Clone the repository
git clone https://github.com/MedicoMrityunjay/DermaSentinel_AI.git
cd DermaSentinel_AI

# 2. Build the Docker image
docker build -t dermasentinel .

# 3. Run the container
docker run -p 7860:7860 dermasentinel
```

### Option 3: Local Development (Python)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python -m uvicorn main_server:app --reload --port 7860
```

---

## ⚠️ Medical Disclaimer

**DermaSentinel is a research tool and is NOT a certified medical device.**
It is intended for **educational and clinical decision support purposes only**. It should never replace the professional judgment of a qualified dermatologist or pathologist. All diagnoses must be verified by standard clinical procedures (dermoscopy, biopsy, histopathology).

---

## 👨‍💻 Credits

**Created by:** Mrityunjay (MedicoMrityunjay)
*   *Lead Developer & Researcher*

**Datasets Used:**
*   ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection
*   ISIC 2019/2020: SIIM-ISIC Melanoma Classification

**License:** MIT License
