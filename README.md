# Papilledema Detection

AI-powered automated diagnosis system using Hybrid Digital Image Processing & Deep Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-94.23%25-brightgreen.svg)
![Recall](https://img.shields.io/badge/Recall-0.96-brightgreen.svg)

## Overview

Papilledema (optic disc swelling) is a critical sign of increased intracranial pressure. However, it looks extremely similar to "Pseudopapilledema" (false swelling), making manual diagnosis error-prone.

This project implements a hybrid system combining **Classical Digital Image Processing** techniques with **Deep Learning** to achieve highly accurate and sensitive diagnosis.

## Key Features

- **Hybrid Preprocessing Pipeline:** CLAHE + Unsharp Masking for enhanced feature visibility
- **Multi-Model Comparison:** MobileNetV2, EfficientNetB0, DenseNet121
- **Safety Mode Training:** Auto-revert mechanism to prevent overfitting
- **Clinical Priority Weighting:** 1.5x boost for Papilledema class to minimize false negatives
- **Grad-CAM Visualization:** Model interpretability and validation
- **Web Interface:** Gradio-based real-time diagnosis demo

## Results

| Model | Accuracy | Recall (Sensitivity) | Status |
|-------|----------|---------------------|--------|
| **MobileNetV2** | **94.23%** | **0.96** | ✅ Winner |
| EfficientNetB0 | 92.79% | 0.89 | Stable |
| DenseNet121 | 91.83% | 0.85 | Overfitted |

> **Surprise Result:** Lightweight MobileNetV2 outperformed heavier architectures due to better adaptation to our preprocessed dataset.

## Methodology

### 1. Preprocessing Pipeline

```
Raw Fundus Image → CLAHE → Unsharp Masking → Model Input
```

- **CLAHE:** Contrast Limited Adaptive Histogram Equalization for revealing vessels
- **Unsharp Masking:** Edge enhancement for optic disc boundaries

### 2. Two-Phase Learning

- **Phase 1 (Transfer Learning):** Frozen base model, training only custom head
- **Phase 2 (Fine-Tuning):** Unfreezing top 20% layers with very low LR (1e-5)

### 3. Safety Mode

Custom callback mechanism that automatically reverts to Phase 1 weights if Phase 2 causes validation loss to increase (overfitting).

## Tech Stack

- Python 3.8+
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib / Seaborn
- Gradio (Web Interface)
- Google Colab (Training)

## Installation

```bash
# Clone the repository
git clone https://github.com/Ozgur492/Papilledema-Detection.git
cd Papilledema-Detection

# Install dependencies
pip install tensorflow opencv-python numpy matplotlib seaborn gradio kagglehub split-folders
```

## Usage

### Training (Google Colab recommended)

```python
# Run the main training script
python dip_fina_project.py
```

### Web Demo

```python
# Launch Gradio interface
python -c "import gradio_demo"
```

## Dataset

Dataset from Kaggle: [Identification of Pseudopapilledema](https://www.kaggle.com/datasets/shashwatwork/identification-of-pseudopapilledema)

**Classes:**
- Normal
- Papilledema
- Pseudopapilledema

**Split:** 70% Train / 15% Validation / 15% Test


## Grad-CAM Visualization

The heatmap validates that the model focuses strictly on the **Optic Disc** and vessel origins, confirming the model is detecting "swelling" and not background artifacts.

## Limitations

1. **Dataset Size:** Small number of samples in "Pseudopapilledema" class
2. **External Validation:** Tested on same source distribution only
3. **Patient Data:** Diagnosis based purely on images, ignoring clinical history


## Team - Papilla Picasso

| Name | Student ID |
|------|------------|
| Doğa Pirci | 24070001119 |
| Gökalp Olukcu | 21070001208 |
| Özgür Can Güngör | 21070001058 |

## 🎓 Course Information

- **Course:** COMP 4360 - Digital Image Processing
- **Instructor:** Dr. Suphi Uçar
- **University:** Yaşar University - Faculty of Engineering


## 📄 License

This project is licensed under the MIT License.

Note: This project was developed for educational purposes

---

<p align="center">
  
</p>
