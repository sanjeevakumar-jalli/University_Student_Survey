import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Import models
from Model.logistic_regression import train_model as train_lr
from Model.decision_tree import train_model as train_dt
from Model.knn import train_model as train_knn
from Model.naive_bayes import train_model as train_nb
from Model.random_forest import train_model as train_rf
from Model.xgboost_model import train_model as train_xgb

# ------------------------------------------
# Streamlit UI
#-------------------------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("World University Student Survey - Classification App")
st.write("Upload test dataset, select a model and view performance metrics.")

# ------------------------------------------------
# Dataset Upload
# ------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file (Test Data Only)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)   
    st.subheader("Clean Dataset Preview")
    st.dataframe(df.head())

    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")
    
    #---------------------------------------------
    # Target Selection
    #---------------------------------------------
    st.subheader("Target Variable Selection")

    target_column = st.selectbox(
        "Select target column",
        options= df.columns
    )

    # Remove rows where target is missing
    df = df.dropna(subset=[target_column]).reset_index(drop=True)

    # Remove rare classes
    class_counts = df[target_column].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df[target_column].isin(valid_classes)].reset_index(drop=True)

    # Safety check
    if df[target_column].nunique() < 2:
        st.error("Target must have at least 2 classes after cleaning.")
        st.stop()
    
    # Separate features and target
    x = df.drop(columns=[target_column])
    y = df[target_column]
    
    #Encode Categorical features
    for col in x.select_dtypes(include=["object"]).columns:
        x[col] = LabelEncoder().fit_transform(x[col])
    
    y = LabelEncoder().fit_transform(y)
    
    #------------------------------------------
    # Train-Test Spilt
    #------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        x,y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X_train)
    x_test = scaler.transform(X_test)
    #------------------------------------------
    #Model Selection
    #------------------------------------------
    st.subheader("Model Selection")
    
    model_name = st.selectbox(
        "Choose Classification Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ]
    )
    
    #---------------------------------------
    # Train Model
    #---------------------------------------
    if model_name == "Logistic Regression":
        model = train_lr(X_train, y_train)
    elif model_name == "Decision Tree":
        model = train_dt(X_train, y_train)
    elif model_name == "KNN":
        model = train_knn(X_train, y_train)
    elif model_name == "Naive Bayes":
        model = train_nb(X_train, y_train)
    elif model_name == "Random Forest":
        model = train_rf(X_train, y_train)
    else:
        model = train_xgb(X_train, y_train)
    
    #--------------------------------------
    # Predictions
    #--------------------------------------
    y_pred = model.predict(X_test)
    
    # AUC Handling
    auc = "NA"
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                auc = roc_auc_score(
                    y_test, y_prob,
                    multi_class="ovr",average="weighted"
                )
        except:
            pass
    
    #----------------------------------------
    #Metrics Display
    #----------------------------------------
    st.subheader("Evaluation Matrics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
    col5.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.3f}")
    col6.metric("AUC", auc if auc == "NA" else f"{auc:.3f}")
    
    # ------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------
    st.subheader("Confusion Matrix")
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    
    # ------------------------------------------------
    # Classification Report
    # ------------------------------------------------
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    







