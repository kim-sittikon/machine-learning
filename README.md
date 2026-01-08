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


This repository demonstrates the practical application of **Supervised Learning** algorithms to solve real-world problems. It is structured into four core modules:

1.  **Neural Networks (`Neural Network/`)**: Focusing on **Multi-layer Perceptrons (MLP)** for classification and regression. Key techniques include dynamic architecture generation, custom image preprocessing pipelines, and recursive time-series forecasting.
2.  **Support Vector Machines (`SVM/`)**: Exploring kernel tricks (RBF, Polynomial, Linear) and their effectiveness on high-dimensional data (Bio-informatics, Medical Imaging).
3.  **Convolutional Neural Networks (`CNN/`)**: Implementation of custom 2D and 1D CNNs for image classification and time-series analysis.
4.  **Deep CNN (`DCNN/`)**: Advanced Transfer Learning using state-of-the-art architectures like VGG16, ResNet50, and DenseNet.

## ✨ Key Features

*   **Comprehensive Coverage**: From basic ML algorithms (SVM) to advanced Deep Learning (Transfer Learning).
*   **Real-world Datasets**: Usage of datasets like LFW (Face Rec), Blood Cells, and COVID-19 timeseries.
*   **Custom Implementations**: Manual implementation of sliding windows, image preprocessing pipelines, and grid searches.
*   **Visual Analytics**: Rich visualizations using Matplotlib and Seaborn for confusion matrices, loss curves, and predictions.

## 📂 Repository Structure

```tree
machine-learning/
├── 🧠 Neural Network/     # MLP Labs (Digits, FaceRec, Iris, Fungi, BloodCells, Covid)
├── 📐 SVM/                # Support Vector Machine Labs
├── 🖼️ CNN/                # Convolutional Neural Network Labs
├── 🚀 DCNN/               # Deep CNN / Transfer Learning Labs
├── 💾 dataset/            # Dataset directory (gitignored)
├── 📄 word/               # Lab Reports and Documentation
└── 📜 README.md           # Project Documentation
```

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
| [**`scikit-learn`**](https://scikit-learn.org/) | The brain of the operation. Used for `MLPClassifier`, `MLPRegressor`, `SVC`, `SVR`, and metrics (`accuracy_score`, `r2_score`). Includes **`joblib`** for model persistence. | 🔴 **Critical** |
| [**`pandas`**](https://pandas.pydata.org/) | Data wrangling for CSV datasets. Handling missing values and rolling windows. *(Note: Requires `openpyxl` if reading .xlsx files)* | 🔴 **Critical** |
| [**`numpy`**](https://numpy.org/) | Low-level matrix operations, especially for flattening images and reshaping arrays. | 🔴 **Critical** |
| [**`seaborn`**](https://seaborn.pydata.org/) | **Statistical Data Visualization**. Used for creating informative Heatmaps (Confusion Matrices) and aesthetic distribution plots. | 🟡 **High** |
| [**`matplotlib`**](https://matplotlib.org/) | Plotting training history, confusion matrices, and forecasting trends. | 🟡 **High** |
| [**`opencv-python`**](https://pypi.org/project/opencv-python/) | Advanced image loading and processing for `cv2` based labs. | 🟡 **High** |
| [**`Pillow` (PIL)**](https://python-pillow.org/) | Alternative lightweight image handling for simpler datasets. | 🟡 **High** |

---

## 🧪 Experiments & Labs

### 🧠 Module 1: Neural Networks (MLP)
*Located in `Neural Network/`*

| Lab | Project Name | Task Type | Advanced Concepts Applied |
| :--- | :--- | :--- | :--- |
| **Lab 01** | [**Handwritten Digits**](Neural%20Network/LAB1_NN_Digits.py) | 🔢 Classification | • Dynamic Hidden Layer Generation<br>• Grid Search for Optimal Nodes |
| **Lab 02** | [**Face Recognition**](Neural%20Network/LAB2_NN_FaceRec.py) | 👤 Vision | • Visual Error Analysis (Bounding Box coloring)<br>• LFW Dataset Handling |
| **Lab 03** | [**Iris Classification**](Neural%20Network/LAB3_NN_Iris.py) | 🌺 Tabular | • Learning Rate Sensitivity Analysis (`1e-2` vs `1e-5`)<br>• Overfitting Detection |
| **Lab 04** | [**Fungi Specimen**](Neural%20Network/LAB4_NN_Fungi.py) | 🍄 Bio-Vision | • Batch Image Processing Pipeline<br>• RGB vs Grayscale Feature Extraction |
| **Lab 05** | [**Blood Cell Analysis**](Neural%20Network/LAB5_NN_BloodCells.py) | 🩸 Medical | • High-Throughput Classification<br>• Architecture Scalability Testing |
| **Lab 06** | [**COVID-19 Forecast**](Neural%20Network/LAB6_NN_Covid_Forecast.py) | 📈 Time-Series | • **Sliding Window Preprocessing** (Seq2Seq equivalent)<br>• Recursive Multi-step Forecasting<br>• Trend vs Actual Visualization |

### 📐 Module 2: Support Vector Machines (SVM)
*Located in `SVM/`*

| Lab | Project Name | Kernel Focus | Description |
| :--- | :--- | :--- | :--- |
| **LEB 01** | [**Iris Species**](SVM/LEB%201%20-%20SVM%20on%20the%20Iris%20Dataset%20(Multiclass%20Classification).py) | Linear/RBF | Baseline comparison of kernel performance on non-linear data. |
| **LEB 03** | [**Flower Vision**](SVM/LEB%203%20-%20SVM%20on%20Flower%20Recognition%20Dataset.py) | Poly | Handling high-dimensional visual data with Polynomial kernels. |
| **LEB 04** | [**Protozoan Parasite**](SVM/LEB%204%20-%20SVM%20on%20Protozoan%20Parasite%20Image%20Data%20(PPID).py) | RBF | Medical image classification with automated feature scaling. |
| **LEB 05** | [**Hemo-Analysis**](SVM/LEB%205%20-%20SVM%20on%20Blood%20Cells%20(microscopic%20peripheral%20blood%20cell%20images).py) | Sigmoid/RBF | Binary/Multi-class classification of blood cell types. |
| **LEB 06** | [**Pandemic Trends**](SVM/LEB%206%20-%20SVM%20with%20Smoothed%20COVID-19%20Data%20(Regression).py) | Regression | Non-linear regression (SVR) for modeling infection rates. |


### 🖼️ Module 3: Convolutional Neural Networks (CNN)
*Located in `CNN/`*

| Lab | Project Name | Type | Key Concepts |
| :--- | :--- | :--- | :--- |
| **Lab 01** | [**Digits Classification**](CNN/LAB1_CNN_Digits.py) | 🔢 Image | • Custom CNN Architectures<br>• Comparative Analysis of Network Depth |
| **Lab 02** | [**Face Recognition**](CNN/LAB2_CNN_FaceRec.py) | 👤 Vision | • LFW Dataset<br>• Complex Image Processing Pipelines |
| **Lab 03** | [**Iris Classification**](CNN/LAB3_CNN_Iris.py) | 🌺 Tabular | • 1D-CNN for Tabular Data<br>• Learning Rate & Batch Size Tuning |
| **Lab 04** | [**Fungi Specimen**](CNN/LAB4_CNN_Fungi.py) | 🍄 Bio-Vision | • Microscopic Image Classification<br>• Network Size Impact Analysis |
| **Lab 05** | [**Blood Cell Analysis**](CNN/LAB5_CNN_BloodCells.py) | 🩸 Medical | • Medical Imaging with CNNs<br>• Automated Cell Type Detection |
| **Lab 06** | [**COVID-19 Forecast**](CNN/LAB6_CNN_COVID.py) | 📈 Time-Series | • 1D-CNN for Time-Series Forecasting<br>• Temporal Pattern Recognition |

### 🚀 Module 4: Deep CNN (Transfer Learning)
*Located in `DCNN/`*

| Lab | Project Name | Models | Description |
| :--- | :--- | :--- | :--- |
| **Lab 01** | [**Advanced Digits**](DCNN/LAB1_DCNN_Digits.py) | VGG, ResNet, DenseNet | Comparative study of pre-trained architectures vs custom models. |

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

