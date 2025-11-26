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

This repository contains code to deploy the DermaSentinel inference engine, including the dual-gate architecture (Segmentation + Classification), AI Scribe, and Equity Engine.

---

## 🚀 Main Findings

DermaSentinel bridges the gap between black-box AI and clinical practice by offering:
1.  **Precision Diagnosis**: A dual-gate architecture (Segmentation + Classification) for high-accuracy lesion analysis.
2.  **Explainability**: Real-time quantification of the **ABCD Rule** (Asymmetry, Border, Color) and visual segmentation overlays.
3.  **Equity**: An integrated **Equity Engine** that analyzes skin tone (ITA Score) to ensure fair performance across diverse skin phenotypes.
4.  **Workflow Integration**: Automated **AI Scribe** for clinical note generation and PDF reporting.

---

## 📦 Dependencies

To clone all files:

```bash
git clone https://github.com/MedicoMrityunjay/DermaSentinel_AI.git
cd DermaSentinel_AI
```

To install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## � Directory Structure

```text
DermaSentinel_AI/
├── core/
│   ├── models/          # PyTorch model definitions (U-Net, EfficientNet)
│   └── database.py      # SQLAlchemy database configuration
├── static/              # Frontend assets (HTML, CSS, JS)
├── main_server.py       # FastAPI backend and inference pipeline
├── Dockerfile           # Container configuration
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## �💾 Data

### Training Datasets
The models were trained on a curated combination of public datasets:
1.  **ISIC 2018 Task 1**: Lesion Boundary Segmentation (2,594 images). Used for training the **Scalpel** (U-Net).
2.  **SIIM-ISIC 2020**: Melanoma Classification (33,126 images). Used for training the **Fusion** (EfficientNet) classifier.

### "Iron Curtain" Split Protocol
To ensure zero patient leakage and robust evaluation, we implemented a **Stratified GroupKFold** split:
*   **Grouping**: Samples were grouped by `patient_id` to ensure images from the same patient never appear in both train and validation sets.
*   **Stratification**: Splits were stratified by target label (benign/malignant) to maintain class balance.

*Note: The raw datasets are not included in this repository due to size and licensing. They can be downloaded from the [ISIC Archive](https://www.isic-archive.com/).*

---

## 🧠 Technical Architecture

DermaSentinel employs a **Dual-Gate** inference pipeline:

### Gate 1: The Scalpel (Segmentation)
*   **Architecture**: U-Net with a ResNet34 encoder (pre-trained on ImageNet).
*   **Input**: 512x512 RGB Image.
*   **Output**: Binary segmentation mask (Lesion vs. Background).
*   **Purpose**: Isolates the lesion for ABCD analysis and calculates the "Mask Coverage" score.

### Gate 2: The Fusion (Classification)
*   **Architecture**: **EfficientNet-B3** (Image Feature Extractor) + **MLP** (Metadata Processor).
*   **Fusion Strategy**: Late fusion. The 1536-dim image embedding is concatenated with a 32-dim metadata embedding (Age, Sex, Site) before the final classification head.
*   **Uncertainty**: Uses **Test-Time Augmentation (TTA)**. The image is passed through the network 5 times with random augmentations (flips, rotations). The final probability is the mean, and uncertainty is the standard deviation.

---

## 💻 Usage

### 1. Running the Server (Docker)
The recommended way to run DermaSentinel is via Docker.

```bash
docker build -t dermasentinel .
docker run -p 7860:7860 dermasentinel
```
Access the UI at `http://localhost:7860`.

### 2. Python API (Inference)
You can also use the models programmatically for batch inference.

```python
import torch
from main_server import scalpel, fusion, base_transforms

# Load Image
image = ... # PIL Image

# Segmentation
input_tensor = base_transforms(image=image_np)["image"].unsqueeze(0).to("cuda")
mask = torch.sigmoid(scalpel(input_tensor))

# Classification
# ... (See main_server.py for full pipeline)
```

---

## 📊 Evaluation & Metrics

The system was evaluated on a held-out test set generated via the Iron Curtain protocol.

| Metric | Value | Description |
| :--- | :--- | :--- |
| **AUC-ROC** | **0.965** | Area Under the Receiver Operating Characteristic Curve |
| **Sensitivity** | **94.2%** | Ability to correctly identify melanoma (True Positive Rate) |
| **Specificity** | **88.5%** | Ability to correctly identify benign lesions (True Negative Rate) |
| **Dice Score** | **0.92** | Segmentation overlap accuracy |
| **Inference** | **< 1.5s** | Average end-to-end processing time on GPU |

---

## 🐛 Issues

Please open new issue threads specifying the issue with the codebase or report issues directly via the GitHub repository.

---

## 📚 Citation

If you use this code or model in your research, please cite:

```bibtex
@software{DermaSentinel2025,
  author = {Mrityunjay},
  title = {DermaSentinel: Clinical-Grade AI for Melanoma Detection},
  year = {2025},
  url = {https://github.com/MedicoMrityunjay/DermaSentinel_AI}
}
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
