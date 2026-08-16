"""
train.py — Train and save all 5 classification models.

Usage:
    python model/train.py

Saves pickle files to model/ directory.
"""

import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV  = os.path.join(BASE_DIR, "train_data.csv")
TEST_CSV   = os.path.join(BASE_DIR, "test_data.csv")
MODEL_DIR  = os.path.dirname(os.path.abspath(__file__))

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]
X_test  = test_df.drop("target", axis=1)
y_test  = test_df["target"]

# ── Scale features ────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Save scaler
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ── Define models ─────────────────────────────────────────────────────────────
models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree":       DecisionTreeClassifier(random_state=42),
    "knn":                 KNeighborsClassifier(n_neighbors=5),
    "naive_bayes":         GaussianNB(),
    "random_forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

# ── Train, evaluate, save ─────────────────────────────────────────────────────
print("\n{:<22} {:>8} {:>8} {:>10} {:>8} {:>8} {:>8}".format(
    "Model", "Acc", "AUC", "Precision", "Recall", "F1", "MCC"))
print("-" * 80)

results = {}
for key, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = (model.predict_proba(X_test_s)[:, 1]
              if hasattr(model, "predict_proba") else y_pred)

    metrics = {
        "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "AUC":       round(roc_auc_score(y_test, y_prob),   4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred),    4),
        "F1":        round(f1_score(y_test, y_pred),        4),
        "MCC":       round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results[key] = metrics

    print("{:<22} {:>8.4f} {:>8.4f} {:>10.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
        key, metrics["Accuracy"], metrics["AUC"], metrics["Precision"],
        metrics["Recall"], metrics["F1"], metrics["MCC"]))

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{key}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved → {model_path}")

print("\nAll models trained and saved successfully.")
