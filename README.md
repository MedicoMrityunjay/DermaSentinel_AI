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

<div align="center">

# 🛡️ DermaSentinel AI
### *Clinical-Grade Melanoma Detection & Analysis System*

[![Status](https://img.shields.io/badge/Status-Gold%20Master%20v3.0-gold?style=for-the-badge&logo=github)](https://github.com/MedicoMrityunjay/DermaSentinel_AI/releases)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/medicomrityunjay/DermaSentinel)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)](https://hub.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.95+-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=flat-square&logo=opencv&logoColor=white"/>
</p>

---

### � **[Launch Live Demo](https://huggingface.co/spaces/medicomrityunjay/DermaSentinel)**
*Experience the full clinical workflow directly in your browser.*

---

</div>

## 🧠 System Architecture: The "Dual-Gate" Pipeline

DermaSentinel employs a novel **Dual-Gate Architecture** that decouples segmentation from classification for maximum explainability and precision.

```mermaid
graph TD
    subgraph Input
    IMG[Input Image] --> PRE[Preprocessing]
    META[Patient Metadata] --> FUSION
    end

    subgraph "Gate 1: The Scalpel 🗡️"
    PRE --> UNET[U-Net (ResNet34)]
    UNET --> MASK[Segmentation Mask]
    MASK --> ABCD[ABCD Analysis]
    end

    subgraph "Gate 2: The Fusion ⚛️"
    PRE --> EFF[EfficientNet-B3]
    EFF --> EMB[Image Embeddings]
    EMB --> FUSION[Late Fusion Layer]
    FUSION --> MLP[Classification Head]
    MLP --> PROB[Malignancy Probability]
    end

    subgraph "Output & Explainability"
    MASK --> VIS[Visual Overlay]
    ABCD --> REP[Clinical Report]
    PROB --> REP
    PROB --> TTA[Uncertainty (TTA)]
    end

    style IMG fill:#f9f,stroke:#333,stroke-width:2px
    style MASK fill:#bbf,stroke:#333,stroke-width:2px
    style PROB fill:#bfb,stroke:#333,stroke-width:2px
```

---

## 🌟 Key Features at a Glance

| Feature | Description | Technology |
| :--- | :--- | :--- |
| **🗡️ The Scalpel** | Pixel-perfect lesion isolation. | **U-Net** + ResNet34 |
| **⚛️ The Fusion** | Multimodal diagnosis (Image + Metadata). | **EfficientNet** + MLP |
| **📝 AI Scribe** | Auto-generates clinical notes. | **BLIP** (Vision-Language) |
| **⚖️ Equity Engine** | Detects skin tone bias (ITA Score). | **CIELab** Color Analysis |
| **🔍 XAI Suite** | Real-time ABCD Rule quantification. | **OpenCV** Geometry |
| **🛡️ Safety Nets** | Blur detection & Uncertainty estimation. | **Laplacian** + **TTA** |

---

## 📊 Clinical Validation Metrics

The system was rigorously evaluated using the **"Iron Curtain"** split protocol (Stratified GroupKFold) to ensure zero patient leakage.

<div align="center">

| Metric | Performance | Clinical Significance |
| :--- | :---: | :--- |
| **AUC-ROC** | **0.965** | Excellent discrimination between benign/malignant. |
| **Sensitivity** | **94.2%** | Minimizes missed melanomas (False Negatives). |
| **Specificity** | **88.5%** | Reduces unnecessary biopsies (False Positives). |
| **Dice Score** | **0.92** | High-fidelity lesion boundary detection. |

</div>

---

## �️ Installation & Deployment

<details>
<summary><b>🐳 Option 1: Docker (Recommended)</b></summary>

```bash
# 1. Clone the repository
git clone https://github.com/MedicoMrityunjay/DermaSentinel_AI.git
cd DermaSentinel_AI

# 2. Build the container
docker build -t dermasentinel .

# 3. Run (Port 7860)
docker run -p 7860:7860 dermasentinel
```
</details>

<details>
<summary><b>🐍 Option 2: Local Python Environment</b></summary>

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python -m uvicorn main_server:app --reload --port 7860
```
</details>

---

## 📂 Repository Structure

```text
DermaSentinel_AI/
├── 🧠 core/
│   ├── models/          # PyTorch Architectures (Scalpel, Fusion)
│   ├── engine/          # Training Loops & Validation
│   └── database.py      # SQLite/SQLAlchemy ORM
├── 🎨 static/           # Frontend (Glassmorphism UI, JS, CSS)
├── 🚀 main_server.py    # FastAPI Inference Engine
├── 🐳 Dockerfile        # Production Container Config
└── 📄 requirements.txt  # Pinned Dependencies
```

---

## ⚖️ Equity & Fairness

DermaSentinel includes a dedicated **Equity Engine** to address racial bias in dermatological AI.

*   **ITA Calculation**: Automatically computes the *Individual Typology Angle* to classify skin phenotype (Fitzpatrick I-VI).
*   **Bias Warning**: If the system detects **Type V or VI** (Dark Skin), it triggers a warning: *"High melanin content detected. Segmentation contrast may be reduced."* This ensures clinicians remain vigilant in underrepresented demographics.

---

## � Citation

```bibtex
@software{DermaSentinel2025,
  author = {Mrityunjay},
  title = {DermaSentinel: Clinical-Grade AI for Melanoma Detection},
  year = {2025},
  url = {https://github.com/MedicoMrityunjay/DermaSentinel_AI}
}
```

---

<div align="center">

### ⚠️ Medical Disclaimer
*DermaSentinel is a research tool for **Educational & Clinical Decision Support** only.*
*It is NOT a diagnostic device. Always verify findings with histopathology.*

**Created by [MedicoMrityunjay](https://github.com/MedicoMrityunjay)**
*Licensed under MIT*

</div>
