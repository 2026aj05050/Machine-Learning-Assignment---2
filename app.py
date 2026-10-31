"""
Dry Bean Classifier - Interactive Evaluation Web App
Loads pre-trained models from disk and evaluates user-uploaded CSV test data.
"""

import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, precision_score, 
    recall_score, f1_score, confusion_matrix, 
    classification_report, roc_auc_score
)

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

st.set_page_config(page_title="Dry Bean Classifier", layout="wide")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
    le = joblib.load(f"{MODEL_DIR}/label_encoder.joblib")
    with open(f"{MODEL_DIR}/feature_columns.json") as f:
        cols = json.load(f)
    models = {name: joblib.load(f"{MODEL_DIR}/{file}") for name, file in MODEL_FILES.items()}
    return scaler, le, cols["features"], cols["target"], models

def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, proba, multi_class='ovr', average='macro'),
        "Precision": precision_score(y, y_pred, average='macro', zero_division=0),
        "Recall": recall_score(y, y_pred, average='macro', zero_division=0),
        "F1 Score": f1_score(y, y_pred, average='macro', zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred)
    }

scaler, label_encoder, feature_columns, target_column, models = load_artifacts()

st.title("Dry Bean Classification Model Evaluator")
st.write(
    "Upload the test CSV file (`test_data.csv`) to evaluate the five trained classifiers "
    "on the UCI Dry Bean dataset. Models are loaded directly from disk without retraining."
)

with st.sidebar:
    st.header("Control Panel")
    chosen_model_name = st.selectbox("Select Model for Detailed View", list(MODEL_FILES.keys()))
    uploaded = st.file_uploader("Upload Test CSV File", type=["csv"])
    st.caption("CSV must contain all 16 feature columns and the 'Class' target column.")

if uploaded is None:
    st.info("Please upload `test_data.csv` using the sidebar to view live model evaluations.")
    st.subheader("Baseline Metrics Recorded During Training")
    st.dataframe(pd.read_csv(f"{MODEL_DIR}/metrics.csv").round(4), use_container_width=True)
    st.stop()

df = pd.read_csv(uploaded)

missing = [col for col in feature_columns + [target_column] if col not in df.columns]
if missing:
    st.error(f"Missing required columns in CSV: {', '.join(missing)}")
    st.stop()

df = df[df[target_column].isin(label_encoder.classes_)]
X = scaler.transform(df[feature_columns].values)
y = label_encoder.transform(df[target_column].values)

st.success(f"Loaded {len(df)} sample rows from the uploaded CSV.")

# 1. Comparison Table across all 5 models
st.subheader("Overall Model Performance Comparison")
table_data = []
for name, model in models.items():
    m = compute_metrics(model, X, y)
    m["Model"] = name
    table_data.append(m)

comp_df = pd.DataFrame(table_data)[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]]
st.dataframe(comp_df.round(4), use_container_width=True)

best_model = comp_df.loc[comp_df["F1 Score"].idxmax(), "Model"]
st.success(f"Best performing model on uploaded dataset (by Macro F1): **{best_model}**")

st.divider()

# 2. Detailed View for Selected Model
st.subheader(f"Detailed Analysis: {chosen_model_name}")
selected_model = models[chosen_model_name]
preds = selected_model.predict(X)
sel_metrics = compute_metrics(selected_model, X, y)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{sel_metrics['Accuracy']:.4f}")
col2.metric("AUC", f"{sel_metrics['AUC']:.4f}")
col3.metric("Precision", f"{sel_metrics['Precision']:.4f}")
col4.metric("Recall", f"{sel_metrics['Recall']:.4f}")
col5.metric("F1 Score", f"{sel_metrics['F1 Score']:.4f}")
col6.metric("MCC", f"{sel_metrics['MCC']:.4f}")

left, right = st.columns(2)

with left:
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax, 
        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

with right:
    st.markdown("### Classification Report")
    report = classification_report(
        y, preds, target_names=label_encoder.classes_, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)