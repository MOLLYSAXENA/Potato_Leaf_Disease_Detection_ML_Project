# 🍃 Potato Leaf Disease Detection Using ResNet-18

An end-to-end computer vision project that leverages transfer learning to classify potato leaf images into **Healthy** or **Diseased (Early Blight)** categories. Built using PyTorch and deployed as an interactive web application.

## 🚀 Live Demo
Check out the interactive web application hosted live on Hugging Face Spaces:
👉 **[https://molly90-potato-disease-detector.hf.space]**

---

## 📊 Project Overview
- **Architecture:** ResNet-18 (Pre-trained on ImageNet)
- **Framework:** PyTorch & Torchvision
- **Dataset:** Slices from the [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- **Optimization:** Adam Optimizer ($lr=0.0001$) with Cross-Entropy Loss
- **Key Techniques:** Data Augmentation (Random Flips/Rotations), Train/Validation Data Splitting, Fine-Tuning.

---

## 📂 Repository Structure
```text
├── plant_disease_detection.ipynb  # Core Colab training & evaluation pipeline
├── app.py                         # Production Gradio web application script
├── requirements.txt               # Required Python packages
└── README.md                      # Project documentation
```
## How to run Locally
### 1. Clone the Repository:
```bash
git clone [https://github.com/MOLLYSAXENA/potato-disease-resnet18.git](https://github.com/MOLLYSAXENA/potato-disease-resnet18.git)
cd potato-disease-resnet18
```
### 2. Install Dependencies:
```bash
pip install -r requirements.txt
```
### 3.Run the Interactive Web UI:
```bash
python app.py
```
## Model Performance & Metrics:
The model utilizes data augmentation on training samples to optimize generalization metrics:

Target Classes: Healthy vs. Diseased
Final Validation Accuracy: ~96.8% to 100% on unseen test splits.

## 👥 Acknowledgments
Dataset provided by the open-source PlantVillage Dataset repository.

Built using the PyTorch ecosystem.
