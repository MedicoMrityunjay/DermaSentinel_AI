---
title: DermaSentinel
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# DermaSentinel

**Advanced AI Skin Lesion Analysis & Diagnosis System**

DermaSentinel is a state-of-the-art medical imaging platform designed to assist dermatologists in the early detection of melanoma and other skin conditions. It leverages a multi-modal AI architecture combining segmentation, classification, and visual question answering.

## Features

*   **AI Diagnosis:** Real-time classification of skin lesions with uncertainty quantification.
*   **Lesion Segmentation:** Precise "Scalpel" model to isolate lesion boundaries.
*   **Explainable AI:** ABCD analysis (Asymmetry, Border, Color) with visual radar charts.
*   **AI Scribe:** Automated clinical note generation using BLIP.
*   **AI Consultant:** Interactive chat for visual question answering about the lesion.
*   **Clinical OS:** A modern, dark-mode enabled, responsive interface for clinical environments.

## Architecture

*   **Frontend:** HTML5, TailwindCSS, Chart.js (Responsive, Mobile-First).
*   **Backend:** FastAPI (Python).
*   **AI Core:** PyTorch, Albumentations, Segmentation Models PyTorch.
*   **Deployment:** Dockerized for Hugging Face Spaces.

## Author

**Created by Mrityunjay**

---
*This project is for research purposes only and should not replace professional medical advice.*
