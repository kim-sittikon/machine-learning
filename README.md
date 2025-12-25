# 🧠 Machine Learning Mastery: Neural Networks & SVM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV%20%2F%20PIL-green)

A comprehensive collection of Machine Learning experiments focusing on **Artificial Neural Networks (MLP)** and **Support Vector Machines (SVM)**. This repository covers data preprocessing, image classification, regression analysis, and time-series forecasting using real-world datasets.

---

## 📂 Project Structure

This repository is divided into two main sections based on the machine learning algorithms used.

### 🕸️ Part 1: Neural Networks (Multi-layer Perceptron)
Experiments using `MLPClassifier` and `MLPRegressor` to solve complex pattern recognition and forecasting tasks. Located in `Neural Network/`.

| File Name | Task | Type | Key Concepts & Techniques |
| :--- | :--- | :--- | :--- |
| `LAB1_NN_Basic_Digits.py` | **Handwritten Digits** | Classification (Top-down) | Architecture comparison (Layers vs Nodes), Pivot Table analysis. |
| `LAB2_NN_Face_Recognition.py` | **Face Recognition** | Image Classification | LFW Dataset, Visualizing predictions (Green/Red boxes). |
| `LAB3_NN_Iris.py` | **Iris Classification** | CSV Data | Learning Rate analysis (10^-2 to 10^-5), Overfitting observations. |
| `LAB4_NN_Fungi.py` | **Microscopic Fungi** | Image Processing | Image flattening, Normalization, RGB conversion, Custom dataset loading. |
| `LAB5_NN_BloodCells.py` | **Blood Cell Classification** | Medical Imaging | Robust image loading, Performance tuning, comparing network sizes. |
| `LAB6_NN_Covid_Forecast.py` | **COVID-19 Forecasting** | Time-Series | **Sliding Window** (Lookback), Recursive Forecasting, Trend prediction. |

### 📐 Part 2: Support Vector Machines (SVM)
Experiments exploring different SVM kernels (Linear, Polynomial, RBF) and hyperparameter tuning. Located in `SVM/`.

| File Name | Task | Type | Description |
| :--- | :--- | :--- | :--- |
| `LEB 1...` | **Iris Classification** | Multiclass | Basic classification using Linear, Poly, and RBF kernels. |
| `LEB 3...` | **Flower Images** | 3D/Complex Data | SVM on image data, handling non-linear patterns. |
| `LEB 4...` | **Protozoan Parasite** | Image Classification | Dynamic kernel selection, Image normalization. |
| `LEB 5...` | **Blood Cells** | Medical Imaging | SVM application on blood cell microscopy images. |
| `LEB 6...` | **COVID-19 SVM** | Regression | SVR (Support Vector Regression) for smoothed COVID-19 cases. |

---

## 🛠️ Technologies & Tools

* **Core Logic:** Python, Scikit-Learn (sklearn)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Image Processing:** OpenCV (`cv2`), Pillow (`PIL`)

## 📊 Key Learnings

1.  **Hyperparameter Tuning:** Analyzing the impact of Learning Rates (`10^-2` to `10^-5`) and Network Architecture (Hidden Layers & Nodes) on model accuracy.
2.  **Data Preprocessing:** 
    * **Images:** Resizing, Flattening (2D to 1D vector), and Min-Max Normalization.
    * **Time-Series:** Creating sequences (Sliding Window technique) for supervised learning.
3.  **Model Evaluation:** Using Accuracy Score, Confusion Matrices, and R2 Score for performance assessment.

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kim-sittikon/machine-learning.git
    cd machine-learning
    ```
2.  **Install dependencies:**
    ```bash
    pip install numpy pandas matplotlib scikit-learn opencv-python pillow
    ```
3.  **Navigate and Run:**
    *   For Neural Networks:
        ```bash
        cd "Neural Network"
        python LAB6_NN_Covid_Forecast.py
        ```
    *   For SVM:
        ```bash
        cd SVM
        python "LEB 6 - SVM with Smoothed COVID-19 Data CSV.py"
        ```

---

### 👨‍💻 Author
**Sittikorn Bunna (Kim)**  
Computer Engineering Student @ RMUTT  
*Exploring the world of AI & Cloud Computing.*
