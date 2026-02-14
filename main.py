import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import(
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

#Import model trainers
from Model.logistic_regression import train_model as train_lr
from Model.decision_tree import train_model as train_dt
from Model.knn import train_model as train_knn
from Model.naive_bayes import train_model as train_nb
from Model.random_forest import train_model as train_rf
from Model.xgboost_model import train_model as train_xgb

pd.set_option("display.max_columns", None)

#----------------Load Dataset---------------------------------
try:
    df = pd.read_csv("data\\world_university_survey_dataset.csv")
except FileNotFoundError:
    raise FileNotFoundError("Dataset file not found. Check path")
except Exception as e:
    raise RuntimeError(f"Error reading dataset: {e}")

if df.empty:
    raise ValueError("Dataset empty")

#-----------------View columns--------------------------------
print("Dataset Shape:", df.shape)
print("Columns", df.columns)


#------------------Target Selection--------------------------
target_column = None
for col in df.columns:
    if df[col].dtype == "object" and df[col].nunique() <= 10:
        target_column = col
        break

if target_column is None:
    raise ValueError("No suitable categorical target column found!")

print("\nSelected Target Column:", target_column)

# Drop missing target values
df = df.dropna(subset=[target_column]).reset_index(drop=True)

if(df[target_column].nunique() < 2):
    raise ValueError("The minimum number of samples in target column is 2")


X = df.drop(columns=[target_column])
y = df[target_column]

#---------------------Encoding-------------------------------
label_encoders = {}
try:
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    y = LabelEncoder().fit_transform(y)
except Exception as e:
    raise RuntimeError(f"Error encoding labels: {e}")

#------------------------Train Test Split--------------------
try:
    X_train, X_test, y_train, y_test = train_test_split(
    X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
    )
except ValueError:
    print("Stratified split failed. Using random split.")
    X_train, X_test, y_train, y_test = train_test_split(
    X,
            y,
            test_size=0.2,
            random_state=42
    )

#----------------Scaling---------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#---------------------Evaluation function---------------------------
def evaluate_model(model):
    y_pred = model.predict(X_test)
    n_classes = len(np.unique(y_test))

    # ---------- AUC HANDLING ----------
    auc = "NA"
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        try:
            if n_classes == 2:
                # Binary classification
                auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                # Multiclass classification
                auc = roc_auc_score(
                    y_test,
                    y_prob,
                    multi_class="ovr",
                    average="weighted"
                )
        except Exception:
            auc = "NA"

    # ---------- OTHER METRICS ----------
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

# ---------------- Train Models ----------------
models = {}

try:
    models["Logistic Regression"] = train_lr(X_train, y_train)
except Exception as e:
    print("Logistic Regression failed:", e)

try:
    models["Decision Tree"] = train_dt(X_train, y_train)
except Exception as e:
    print("Decision Tree failed:", e)

try:
    models["KNN"] = train_knn(X_train, y_train)
except Exception as e:
    print("KNN failed:", e)

try:
    models["Naive Bayes"] = train_nb(X_train, y_train)
except Exception as e:
    print("Naive Bayes failed:", e)

try:
    models["Random Forest"] = train_rf(X_train, y_train)
except Exception as e:
    print("Random Forest failed:", e)

try:
    models["XGBoost"] = train_xgb(X_train, y_train)
except Exception as e:
    print("XGBoost failed:", e)

# ---------------- Results ----------------
results = []
for name, model in models.items():
    metrics = evaluate_model(model)
    metrics["Model"] = name
    results.append(metrics)

results_df = pd.DataFrame(results)
print("\n=== MODEL COMPARISON TABLE ===\n")
print(results_df)
