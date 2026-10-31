"""
Dry Bean Classifier - Training Script
Trains 5 classification models on the UCI Dry Bean dataset, evaluates all 6 metrics,
and serializes artifacts into the 'model/' directory.
"""

import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, precision_score, 
    recall_score, f1_score, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV

DATA_FILE = "data/dry_bean.xlsx"
MODEL_DIR = "model"
TARGET = "Class"
SEED = 17

def load_data():
    df = pd.read_excel(DATA_FILE)
    return df.dropna().reset_index(drop=True)

def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED, max_depth=12),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": CalibratedClassifierCV(SVC(random_state=SEED), ensemble=False)
    }

def score_model(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    auc = roc_auc_score(y_test, proba, multi_class='ovr', average='macro')

    return {
        "Accuracy": accuracy_score(y_test, preds),
        "AUC": auc,
        "Precision": precision_score(y_test, preds, average='macro', zero_division=0),
        "Recall": recall_score(y_test, preds, average='macro', zero_division=0),
        "F1 Score": f1_score(y_test, preds, average='macro', zero_division=0),
        "MCC": matthews_corrcoef(y_test, preds)
    }

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()
    feature_cols = [col for col in df.columns if col != TARGET]
    print(f"Dataset Loaded: {len(df)} rows | {len(feature_cols)} features | {df[TARGET].nunique()} classes")

    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Scale all features uniformly across all models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = build_models()
    rows = []

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        metrics = score_model(model, X_test_scaled, y_test)
        metrics["Model"] = name
        rows.append(metrics)

        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(MODEL_DIR, fname))
        print(f"{name:<24} Acc={metrics['Accuracy']:.4f} AUC={metrics['AUC']:.4f} F1={metrics['F1 Score']:.4f} MCC={metrics['MCC']:.4f}")

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump({"features": feature_cols, "target": TARGET}, f, indent=2)

    results = pd.DataFrame(rows)[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]]
    results.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)

    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df[TARGET] = le.inverse_transform(y_test)
    test_df.to_csv("test_data.csv", index=False)
    print(f"\nSaved {len(test_df)} rows to test_data.csv")

if __name__ == "__main__":
    main()