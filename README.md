# 🧠 Machine Learning & SVM Mastery

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

> A collection of Machine Learning experiments focusing on **Support Vector Machines (SVM)** for Classification and Regression tasks. This repository covers data processing, image classification, and time-series forecasting using real-world datasets.

---

## 📂 Project Structure

This repository contains 6 key laboratories exploring different capabilities of SVM kernels (Linear, Poly, RBF).

| Lab | Project Name | Type | Description |
| :---: | :--- | :---: | :--- |
| 01-02 | **Iris Classification** | `CSV Data` | Basic classification on the classic Iris dataset using varying SVM kernels. |
| 03 | **Iris Image Vision** | `Comp Vision` | Image processing & classification of Iris flowers using flattened pixel arrays. |
| 04 | **Protozoan Parasite** | `Medical AI` | Classifying microscopic images of protozoan parasites with confusion matrix analysis. |
| 05 | **Blood Cells Detection** | `Medical AI` | detecting 4 types of white blood cells (Eosinophil, etc.) with auto-directory scanning. |
| 06 | **COVID-19 Forecasting** | `Time Series` | Predicting new cases in Thailand using **SVR (Support Vector Regression)** with sliding window technique. |

---

## 🛠️ Tech Stack

* **Language:** Python
* **Machine Learning:** Scikit-Learn (SVC, SVR)
* **Image Processing:** OpenCV (cv2)
* **Data Manipulation:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn

---

## 🚀 Key Features

### 🌸 Image Classification (Labs 3-5)
- **Pipeline:** Load Image $\rightarrow$ Grayscale $\rightarrow$ Resize (64x64) $\rightarrow$ Flatten $\rightarrow$ SVM.
- **Comparison:** Evaluated `Linear`, `Polynomial`, and `RBF` kernels to find the best accuracy.
- **Visualization:** Includes sample predictions (Green=Correct, Red=Incorrect) and Confusion Matrices.

### 📈 Time-Series Forecasting (Lab 6)
- **Method:** Sliding Window approach (using past 14 days to predict the next day).
- **Model:** Support Vector Regression (SVR).
- **Output:** Recursive forecasting for the next **90 days** (3 months).

---

## 📊 Sample Results

### Medical Image Classification (Blood Cells)
*(Place your screenshot of the Blood Cell prediction/Confusion Matrix here)*
![Blood Cell Result](./images/sample_blood_cell.png)

### COVID-19 SVR Forecast
*(Place your screenshot of the COVID Graph here)*
![COVID Graph](./images/covid_forecast.png)

---

## 💻 How to Run

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/kim-sittikon/machine-learning.git](https://github.com/kim-sittikon/machine-learning.git)
    cd machine-learning
    ```

2.  **Install Dependencies**
    ```bash
    pip install numpy pandas matplotlib scikit-learn opencv-python
    ```

3.  **Setup Datasets**
    * Download datasets from Kaggle (links provided in each Lab file).
    * Or use the provided `setup_dataset.py` script to organize files from your Downloads folder automatically.

4.  **Run a Lab**
    ```bash
    python lab06_covid_forecasting.py
    ```

---

## 👨‍💻 Author

**Kim Sittikon**
* Computer Engineering Student @ RMUTT CPE 🇹🇭
* Interests: Web Development, Cloud Computing, Cybersecurity, and AI.

---
*Created for Machine Learning Course 2025.*
