<div align="center">

# 🧠 Machine Learning Mastery
### Advanced Neural Networks (MLP) & Support Vector Machines (SVM)

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

*A production-grade collection of machine learning experiments, featuring deep dives into MLP architectures, hyperparameter tuning, computer vision pipelines, and time-series forecasting.*

[Report Bug](https://github.com/kim-sittikon/machine-learning/issues) · [Request Feature](https://github.com/kim-sittikon/machine-learning/issues)

</div>

---

## 🎯 Project Overview

This repository demonstrates the practical application of **Supervised Learning** algorithms to solve real-world problems. It is structured into two core modules:

1.  **Neural Networks (`Neural Network/`)**: Focusing on **Multi-layer Perceptrons (MLP)** for classification and regression. Key techniques include dynamic architecture generation, custom image preprocessing pipelines, and recursive time-series forecasting.
2.  **Support Vector Machines (`SVM/`)**: Exploring kernel tricks (RBF, Polynomial, Linear) and their effectiveness on high-dimensional data (Bio-informatics, Medical Imaging).

---

## ⚡ Prerequisites & Installation

To run these labs successfully, you need a standard Data Science environment.

### 1. Core Dependencies
The project relies on the following libraries. Install them via `pip`:

```bash
# Core Data Science Stack
pip install numpy pandas matplotlib seaborn

# Machine Learning & Computer Vision
pip install scikit-learn opencv-python pillow

# Optional (for legacy Excel files)
pip install openpyxl
```

### 2. Dependency Breakdown
| Library | Usage | Importance |
| :--- | :--- | :--- |
| **`scikit-learn`** | The brain of the operation. Used for `MLPClassifier`, `MLPRegressor`, `SVC`, `SVR`, and metrics (`accuracy_score`, `r2_score`). Includes **`joblib`** for model persistence. | 🔴 **Critical** |
| **`pandas`** | Data wrangling for CSV datasets. Handling missing values and rolling windows. *(Note: Requires `openpyxl` if reading .xlsx files)* | 🔴 **Critical** |
| **`numpy`** | Low-level matrix operations, especially for flattening images and reshaping arrays. | 🔴 **Critical** |
| **`seaborn`** | **Statistical Data Visualization**. Used for creating informative Heatmaps (Confusion Matrices) and aesthetic distribution plots. | 🟡 **High** |
| **`matplotlib`** | Plotting training history, confusion matrices, and forecasting trends. | 🟡 **High** |
| **`opencv-python`** | Advanced image loading and processing for `cv2` based labs. | 🟡 **High** |
| **`Pillow` (PIL)** | Alternative lightweight image handling for simpler datasets. | 🟡 **High** |

---

## 🧪 Experiments & Labs

### 🧠 Module 1: Neural Networks (MLP)
*Located in `Neural Network/`*

| Lab | Project Name | Task Type | Advanced Concepts Applied |
| :--- | :--- | :--- | :--- |
| **Lab 01** | **Handwritten Digits** | 🔢 Classification | • Dynamic Hidden Layer Generation<br>• Grid Search for Optimal Nodes |
| **Lab 02** | **Face Recognition** | 👤 Vision | • Visual Error Analysis (Bounding Box coloring)<br>• LFW Dataset Handling |
| **Lab 03** | **Iris Classification** | 🌺 Tabular | • Learning Rate Sensitivity Analysis (`1e-2` vs `1e-5`)<br>• Overfitting Detection |
| **Lab 04** | **Fungi Specimen** | 🍄 Bio-Vision | • Batch Image Processing Pipeline<br>• RGB vs Grayscale Feature Extraction |
| **Lab 05** | **Blood Cell Analysis** | 🩸 Medical | • High-Throughput Classification<br>• Architecture Scalability Testing |
| **Lab 06** | **COVID-19 Forecast** | 📈 Time-Series | • **Sliding Window Preprocessing** (Seq2Seq equivalent)<br>• Recursive Multi-step Forecasting<br>• Trend vs Actual Visualization |

### 📐 Module 2: Support Vector Machines (SVM)
*Located in `SVM/`*

| Lab | Project Name | Kernel Focus | Description |
| :--- | :--- | :--- | :--- |
| **LEB 01** | **Iris Species** | Linear/RBF | Baseline comparison of kernel performance on non-linear data. |
| **LEB 03** | **Flower Vision** | Poly | Handling high-dimensional visual data with Polynomial kernels. |
| **LEB 04** | **Protozoan Parasite** | RBF | Medical image classification with automated feature scaling. |
| **LEB 05** | **Hemo-Analysis** | Sigmoid/RBF | Binary/Multi-class classification of blood cell types. |
| **LEB 06** | **Pandemic Trends** | Regression | Non-linear regression (SVR) for modeling infection rates. |

---

## 🚀 Quick Start Guide

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/kim-sittikon/machine-learning.git
    cd machine-learning
    ```

2.  **Run a Specific Lab (e.g., COVID-19 Forecasting)**
    ```bash
    cd "Neural Network"
    python LAB6_NN_Covid_Forecast.py
    ```

3.  **View Results**
    *   **Console**: Accuracy scores, confusion matrices, and R2 scores will be printed.
    *   **Visuals**: Matplotlib/Seaborn windows will pop up showing prediction graphs and trend comparisons.

---

## 👨‍💻 Author Profile

**Sittikorn Bunna (Kim)**
*Computer Engineering Student @ RMUTT*

<div align="left">
  <a href="https://github.com/kim-sittikon">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />
  </a>
</div>

