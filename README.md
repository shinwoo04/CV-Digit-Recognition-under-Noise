# Robust Handwritten Digit Recognition under Structured Noise
> **A Study on Deep Learning Models for Robust Digit Classification in Occluded Environments**

## 1. Project Overview
This project focuses on developing a robust deep learning model to classify handwritten digits (0-9) occluded by random alphabetic character overlays. Unlike standard digit recognition tasks, this environment introduces **structured noise**, necessitating a model that can effectively disentangle target features from complex background interference. The primary objective is to maximize **model robustness** and **generalization performance** for real-world noisy data.

## 2. Problem Statement & Analysis
* **The Challenge:** High-intensity background letters overlap with target digits, making traditional image processing or simple CNNs insufficient for accurate separation.
* **Exploratory Data Analysis (EDA):** In `notebooks/01_Data_Inference_Visualization.ipynb`, I analyzed the pixel intensity distributions, confirming that the noise and signal occupy the same feature space.
* **Error Analysis:** Using `notebooks/02_Error_Analysis_v2.ipynb`, I identified specific failure modes where the model confuses digits with similar structural components of background letters (e.g., '0' vs. 'Q').



## 3. Experimental Roadmap & Results

| Version | Model Architecture | Preprocessing & Augmentation | Public Acc. | Private Acc. |
|:---:|:---:|:---|:---:|:---:|
| **v1** | ResNet18 (Baseline) | Basic Resize (224x224) | 0.7941 | 0.7587 |
| **v2** | **ResNet50** | **Advanced Augmentation** | **0.8382** | **0.8587** |


### [Key Technical Insights]
* **Model Capacity:** Upgrading to **ResNet50** provided the necessary depth to learn complex hierarchical features required to filter out structured noise.
* **Data Augmentation:** By implementing `RandomRotation`, `RandomAffine`, and `ColorJitter`, the model learned to focus on the invariant structural properties of digits rather than the noise orientation.
* **Generalization Success:** In v2, the **Private Score (0.8587) outperformed the Public Score (0.8382)**. This empirical evidence demonstrates that my augmentation strategy effectively prevented overfitting and ensured high reliability on unseen data—a critical factor for graduate-level research.

## 4. Directory Structure
```text
├── data/               # Raw datasets (Excluded via .gitignore)
├── models/             # Trained model weights (.pth)
├── notebooks/          
│   ├── 01_Data_Inference_Visualization.ipynb  # EDA and problem definition
│   └── 02_Error_Analysis_v2.ipynb             # Systematic failure analysis
├── main.py             # Main training script 
├── inference.py        # Inference and submission generation
├── notes.txt           # Detailed experimental logs and insights
└── requirements.txt    # Environment specifications for reproducibility
