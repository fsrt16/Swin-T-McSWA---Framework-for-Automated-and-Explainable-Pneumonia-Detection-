# Swin-T-McSWA- -Framework-for-Automated-and-Explainable-Pneumonia-Detection-
A Temporal Space-Driven Multi-Context Shifted Window Swin Transformer (Swin-T-McSWA) Framework for Automated and Explainable Pneumonia Detection in the Pulmonary Alveolar Region


# Swin-T-McSWA: Shifted Window Transformer with Multi-context Swin Attention

![Architecture Diagram](Images/Picture2.png)
> **Figure 1:** Swin-T-McSWA Model Architecture

---

## 🧠 Overview

**Swin-T-McSWA** (Shifted Window Transformer with Multi-context Swin Attention) is a hybrid neural architecture that combines the hierarchical feature encoding capabilities of **DenseNet169**, with a novel **T-block** for convolutional attention, followed by **Swin Transformer-inspired window-based multi-head attention** for global context modeling.

This architecture is particularly effective for medical image classification and segmentation tasks, achieving **state-of-the-art (SOTA)** performance while maintaining a relatively small computational footprint.

---

## 📊 Key Contributions

- ✅ Integration of **DenseNet169** with a **custom T-block** for localized receptive field enhancement.
- ✅ Use of **shifted window multi-head self-attention** inspired by Swin Transformers to learn hierarchical context.
- ✅ Lightweight and fast inference design with only **37M parameters** and **7.2 GFLOPs**.
- ✅ Interpretable decision-making via **XAI visualizations** and attention heatmaps.

---

## 🧪 Model Architecture Summary

| Layer Type                   | Output Shape         | Param Count     |
|-----------------------------|----------------------|-----------------|
| DenseNet169 (frozen)        | (5, 5, 1664)         | 12.6M           |
| T-block (dilated + conv + attention) | (5, 5, 1664)         | ~49.8M          |
| Transformer (2× blocks)     | (25, 64)             | ~149K           |
| Classifier Head             | (Dense layers)       | ~17K            |
| **Total Parameters**        |                      | **63.4M**       |
| **Trainable Parameters**    |                      | **50.8M**       |
| **FLOPs**                   |                      | **7.2G**        |

---

## 📈 Visual Workflow

![Training Pipeline](Images/Picture1.png)
> **Figure 2:** Workflow and data pipeline of the Swin-T-McSWA training and evaluation.

---

## 🧪 Sample Input

![Sample Input](Images/DevPneumonia-Page-2.drawio.png)
> **Figure 3:** Example pneumonia X-ray data used in training and evaluation.

---

## 🔍 Explainable AI (XAI)

### 🔦 Attention Heatmap Visualization (I)

![XAI Heatmap 1](Images/DevPneumonia-Page-3.drawio.png)
> **Figure 4:** First-layer attention mapping for model explainability using Grad-CAM and Swin attention outputs.

### 🔦 Attention Heatmap Visualization (II)

![XAI Heatmap 2](Images/Pneumonia.drawio.png)
> **Figure 5:** Deep attention fusion patterns highlighting class-specific saliency.

---

## 🔬 Ablation Study Results

| Configuration Comparison                                  | Params (M) | FLOPs (G) | Accuracy (%) | F1-Score (%) |
|-----------------------------------------------------------|------------|-----------|--------------|--------------|
| Hierarchical vs. Non-Hierarchical Attention               | 37         | 7.2       | 98.47        | 98.03        |
| Shifted Window Attention vs. Standard Multi-head Self-Att | 37         | 7.2       | 98.46        | 98.01        |
| Window Size (2×2 vs. 4×4 vs. 8×8)                         | 37         | 7.2       | 98.50        | 98.08        |
| Self-Attention vs. Cross-Attention                        | 37         | 7.2       | 98.35        | 97.98        |
| Positional Encoding: With vs. Without                     | 37         | 7.2       | 98.40        | 98.00        |

---

## 🚀 SOTA Comparison

| Model                        | Accuracy (%) | F1-Score (%) | Parameters (M) | FLOPs (G) |
|-----------------------------|--------------|--------------|----------------|-----------|
| **Swin-T-McSWA (Proposed)** | **98.76**    | **98.17**    | 37             | 7.2       |
| ViT-Large                   | 94.3         | 94.2         | 307            | 60+       |
| DenseNet201                 | 94.5         | 94.6         | -              | -         |
| EfficientFormer-L1         | 94.3         | 94.4         | 12             | 1.3       |
| MobileViT                   | 93.9         | 93.8         | 5.5            | 0.7       |
| BEiT                        | 94.2         | 94.0         | 86             | 17.6      |
| Vision Transformer (ViT)   | 97.61        | 95.00        | -              | -         |

---

## 📦 Repository Structure

```bash
├── Images/
│   ├── Picture1.png               # Flow diagram
│   ├── Picture2.png               # Model architecture
│   ├── DevPneumonia-Page-2.drawio.png  # Sample input
│   ├── DevPneumonia-Page-3.drawio.png  # XAI heatmap 1
│   ├── Pneumonia.drawio.png           # XAI heatmap 2
├── swin_t_mcswa_model.py         # Model implementation
├── train.py                      # Training script
├── inference.py                  # Inference/Prediction
├── requirements.txt
└── README.md                     # This file


🔧 Setup and Usage
bash
Copy
Edit
# Clone the repo
git clone https://github.com/your-name/swin-t-mcswa.git
cd swin-t-mcswa

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py --dataset_path ./data/ --epochs 100

# Inference
python inference.py --input ./test_image.png
🧠 Citation
bibtex
Copy
Edit
@article{banerjee2025swinmcswa,
  title={Swin-T-McSWA: Shifted Window Transformer with Multi-context Swin Attention for Medical Image Classification},
  author={Banerjee, Tathagat},
  journal={arXiv preprint arXiv:2506.xxxxx},
  year={2025}
}


