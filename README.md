# Dry Bean Classification — BITS Pilani Machine Learning Assignment 2

## 1. Problem Statement
The goal of this project is to automate the classification of seven distinct registered varieties of dry beans (`BARBUNYA`, `BOMBAY`, `CALI`, `DERMASON`, `HOROZ`, `SEKER`, and `SIRA`) based on 16 computer-vision-extracted morphological and geometric features. 

Accurate classification using machine learning enables high-throughput quality control, seed certification, and commercial grading without manual, labor-intensive inspection. This end-to-end pipeline evaluates five supervised machine learning models on a stratified 80/20 split, serializes trained models and preprocessing artifacts, and provides an interactive Streamlit web application for real-time model evaluation using held-out test data.

---

## 2. Dataset Description
* **Dataset Name:** UCI Dry Bean Dataset
* **Source:** UCI Machine Learning Repository / OpenML
* **Instances:** 13,611 physical grain samples
* **Features:** 16 numeric continuous features (extracted from high-resolution grain images):
  1. `Area`: Area of bean zone ($A$)
  2. `Perimeter`: Perimeter of bean ($P$)
  3. `MajorAxisLength`: Longest distance ($L$)
  4. `MinorAxisLength`: Smallest distance ($l$)
  5. `AspectRation`: Aspect ratio ($K = L / l$)
  6. `Eccentricity`: Eccentricity of fitted ellipse
  7. `ConvexArea`: Area of smallest convex polygon containing bean
  8. `EquivDiameter`: Equivalent diameter ($\sqrt{4A / \pi}$)
  9. `Extent`: Ratio of pixel area to bounding box area
  10. `Solidity`: Ratio of area to convex area
  11. `Roundness`: Roundness degree ($4 \pi A / P^2$)
  12. `Compactness`: Measure of roundness ($E = \sqrt{4A/\pi} / L$)
  13. `ShapeFactor1`: Morphological shape metric 1
  14. `ShapeFactor2`: Morphological shape metric 2
  15. `ShapeFactor3`: Morphological shape metric 3
  16. `ShapeFactor4`: Morphological shape metric 4
* **Target Class:** 7 categories (`BARBUNYA`, `BOMBAY`, `CALI`, `DERMASON`, `HOROZ`, `SEKER`, `SIRA`).

---

## 3. Mandatory Links
* **GitHub Public Repository:** [https://github.com/2026aj05050/Machine-Learning-Assignment---2-](https://github.com/2026aj05050/Machine-Learning-Assignment---2-)
* **Live Deployed Streamlit App:** [https://ngjbaxjniekgeznf3siybr.streamlit.app/](https://ngjbaxjniekgeznf3siybr.streamlit.app/)
* **Held-Out Test Data File:** `test_data.csv` (located in the repository root directory)

---

## 4. Model Evaluation & Metrics

All models were evaluated on the exact same 20% held-out test set ($N = 2,723$) using a fixed random seed (`SEED = 17`). Standard scaling (`StandardScaler`) was fitted exclusively on the training set.

### Performance Comparison Table

| Model | Accuracy | AUC (Macro OvR) | Precision (Macro) | Recall (Macro) | F1 Score (Macro) | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9251 | 0.9935 | 0.9324 | 0.9231 | 0.9275 | 0.9102 |
| **Decision Tree** | 0.8950 | 0.9372 | 0.8981 | 0.8964 | 0.8972 | 0.8741 |
| **K-Nearest Neighbors (KNN)** | 0.9240 | 0.9856 | 0.9301 | 0.9215 | 0.9256 | 0.9088 |
| **Naive Bayes (GaussianNB)** | 0.8975 | 0.9870 | 0.9015 | 0.8988 | 0.8995 | 0.8772 |
| **Support Vector Machine (SVM)** | **0.9302** | **0.9948** | **0.9365** | **0.9288** | **0.9325** | **0.9163** |

---

## 5. Performance Observations & Comparison

| Model | Key Strengths & Behavioral Analysis | Limitations / Weaknesses |
| :--- | :--- | :--- |
| **Logistic Regression** | Strong baseline performance (92.51% accuracy); benefits from feature standardization across linear boundaries. High macro AUC (0.9935). | Slightly underperforms SVM on subtle non-linear boundaries between overlapping classes (`DERMASON` vs `SIRA`). |
| **Decision Tree** | Fast, intuitive, and non-parametric. High interpretability with explicit decision boundaries. | Prone to local variance and overfitting; lowest overall accuracy (89.50%) and MCC (0.8741). |
| **K-Nearest Neighbors (KNN)** | Robust distance-based performance (92.40% accuracy) as morphological metrics form tight spatial clusters. | Inference computational cost scales with dataset size; sensitive to noise in boundaries. |
| **Naive Bayes (GaussianNB)** | Computationally efficient probabilistic classifier; solid macro AUC (0.9870). | Feature independence assumption is partially violated due to high correlation among dimension features. |
| **Support Vector Machine (SVM)** | **Top Overall Performer.** RBF kernel effectively models non-linear feature interactions, achieving 93.02% accuracy and 0.9163 MCC. | High calibration overhead during probability estimation. |

### Overall Winner Selection
**Support Vector Machine (SVM)** is selected as the best overall model. It achieves top performance across all six evaluation metrics:
* Highest **Accuracy** ($93.02\%$)
* Highest **Macro OvR AUC** ($0.9948$)
* Highest **F1 Score** ($0.9325$)
* Highest **Matthews Correlation Coefficient (MCC)** ($0.9163$)

---

## 6. Repository Directory Structure

```text
Machine-Learning-Assignment---2-/
├── app.py                      # Streamlit web application
├── train_models.py             # Script to train, evaluate, and serialize models
├── requirements.txt            # Project dependencies
├── README.md                   # Assignment documentation and comparison metrics
├── test_data.csv               # 20% held-out test dataset (raw features + label)
├── data/
│   └── Dry_Bean_Dataset.xlsx   # Raw input dataset
└── model/                      # Serialized model artifacts
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── scaler.joblib
    ├── label_encoder.joblib
    ├── feature_columns.json
    └── metrics.csv


## 7. Local Setup & Execution Instructions

### Prerequisites
* Python 3.9 or higher

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone [https://github.com/2026aj05050/Machine-Learning-Assignment---2-.git](https://github.com/2026aj05050/Machine-Learning-Assignment---2-.git)
cd Machine-Learning-Assignment---2-

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Model Training (Optional)
To retrain the models and regenerate serialized artifacts in `model/`:
```bash
python train_models.py
```

### Step 4: Run Streamlit Web Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser, choose a model from the sidebar dropdown, and upload `test_data.csv` to perform live batch evaluation.