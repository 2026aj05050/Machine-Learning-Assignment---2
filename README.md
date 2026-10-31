# Dry Bean Classification – ML Assignment 2

This repo contains my submission for Machine Learning Assignment 2. I trained five classification models on the UCI Dry Bean dataset and put them behind a small Streamlit app so the results can be checked on any test file without retraining.

## a. Problem statement

Given a set of shape and size measurements taken from images of dried beans, predict which of the seven bean varieties a sample belongs to. This is a multi-class classification problem (7 classes). The practical use is automatic sorting of bean seeds for quality control instead of doing it by hand.

## b. Dataset description

- **Source:** UCI Machine Learning Repository – Dry Bean Dataset (https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Instances:** 13,611
- **Features:** 16 numeric features (12 dimensional measurements + 4 shape factors): Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1–4
- **Target:** `Class` – one of BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA
- **Missing values:** none

The classes are a bit imbalanced (DERMASON has 3,546 samples while BOMBAY only has 522), so I looked at macro-averaged metrics as well as accuracy. All features are continuous, so I standardised them with `StandardScaler` before training the models that are sensitive to feature scale (logistic regression, KNN and SVM). I kept 20% of the data as a stratified test split, which is the `test_data.csv` used by the app.

## c. GitHub repository link

<!-- paste your repository URL here after you push the code -->
https://github.com/<your-username>/dry-bean-classification

## d. Models used

All five models were trained on the same 80/20 stratified split. AUC is the macro one-vs-rest ROC AUC, and precision/recall/F1 are macro-averaged so every bean class counts equally.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9192 | 0.9946 | 0.9321 | 0.9311 | 0.9316 | 0.9023 |
| Decision Tree | 0.9019 | 0.9541 | 0.9130 | 0.9130 | 0.9130 | 0.8815 |
| KNN | 0.9203 | 0.9881 | 0.9330 | 0.9307 | 0.9316 | 0.9037 |
| Naive Bayes | 0.8895 | 0.9909 | 0.8972 | 0.9007 | 0.8988 | 0.8666 |
| Support Vector Machine | 0.9258 | 0.9953 | 0.9380 | 0.9350 | 0.9364 | 0.9103 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Surprisingly strong for a linear model – 91.9% accuracy and the second highest AUC (0.9946). The classes separate fairly well once the features are scaled, so a linear boundary already does most of the job. |
| Decision Tree | Weakest of the five. It still reaches 90% accuracy but its AUC (0.9541) is well below the rest because a single tree gives hard, less calibrated probabilities. I capped the depth at 12 to stop it overfitting. |
| KNN | Very close to logistic regression (92.0% accuracy). With scaled features k=7 neighbours works nicely, though it is the slowest at prediction time and the model file is large since it stores the training set. |
| Naive Bayes | Lowest accuracy and MCC. The Gaussian assumption and the fact that features like Area and ConvexArea are strongly correlated hurt it, but it is extremely fast to train and still keeps a high AUC (0.9909). |
| Support Vector Machine | Best overall on every metric – 92.6% accuracy, 0.9953 AUC and 0.9103 MCC. The RBF kernel handles the non-linear boundaries between similar bean types (SIRA vs DERMASON) better than the others. |
| **Overall, winner for this dataset?** | **Support Vector Machine** – it tops accuracy, AUC, precision, recall, F1 and MCC, though logistic regression and KNN are close enough to be good lightweight alternatives. |

## Project structure

```text
.
├── app.py                     # Streamlit app (loads models, evaluates uploaded CSV)
├── train_models.py            # trains the 5 models and saves everything
├── requirements.txt
├── test_data.csv              # held-out 20% test split (upload this in the app)
├── data/
│   └── dry_bean.xlsx          # raw dataset from UCI
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    ├── scaler.joblib
    ├── label_encoder.joblib
    ├── feature_columns.json
    └── metrics.csv
```

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py         # only needed to regenerate the model files
streamlit run app.py
```

Then open the local URL, pick a model in the sidebar and upload `test_data.csv`.

## About the app

The Streamlit app:

- lets you upload a test CSV (only the test data, as suggested in the instructions),
- has a dropdown to choose any of the five models,
- shows all six evaluation metrics for the selected model and a comparison table for
  all five models at once,
- draws the confusion matrix and the full classification report.

Important: the app **loads the already-trained models from the `model/` folder and
only evaluates** the uploaded data. It never retrains, which is why the metrics stay
consistent with what `train_models.py` produced.

## Live app

<!-- paste your Streamlit Community Cloud URL here after you deploy -->
https://<your-app-name>.streamlit.app