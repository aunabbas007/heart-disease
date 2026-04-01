# ============================================================
# Heart Disease Prediction — SVM Model
# model.py
# ============================================================
# Usage:
#   Train  : python model.py --mode train --data heart.csv
#   Predict: python model.py --mode predict --data new_data.csv
# ============================================================

import argparse
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────
MODEL_PATH  = "svm_heart_model.pkl"
SCALER_PATH = "svm_heart_scaler.pkl"

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]
TARGET_COL = "target"


# ── 1. Load & validate data ───────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {df.shape[0]} rows × {df.shape[1]} cols from '{path}'")
    return df


# ── 2. Preprocess ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame, fit_scaler: bool = True,
               scaler: StandardScaler = None):
    """
    Splits features/target, scales features.
    fit_scaler=True  → fit a new scaler (training time)
    fit_scaler=False → use the provided scaler (inference time)
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy() if TARGET_COL in df.columns else None

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("Must provide a fitted scaler when fit_scaler=False")
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler


# ── 3. Train ──────────────────────────────────────────────────
def train(data_path: str):
    df = load_data(data_path)

    # Basic EDA
    print(f"\n[INFO] Class distribution:\n{df[TARGET_COL].value_counts().to_string()}")
    print(f"[INFO] Missing values: {df.isnull().sum().sum()}")

    X_scaled, y, scaler = preprocess(df, fit_scaler=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Model ──────────────────────────────────────────────────
    # kernel='rbf'   : handles non-linear boundaries via kernel trick
    # C=1.0          : regularization — balances margin width vs misclassifications
    # gamma='scale'  : 1 / (n_features * X.var()), a safe default
    model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42, probability=True)
    model.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*45}")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"{'='*45}")
    print(classification_report(y_test, y_pred,
                                 target_names=["No Disease", "Disease"]))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("SVM — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("[INFO] Saved confusion_matrix.png")
    plt.show()

    # Kernel comparison
    print("\n[INFO] Kernel comparison:")
    for k in ["linear", "rbf", "poly", "sigmoid"]:
        m = SVC(kernel=k, random_state=42)
        m.fit(X_train, y_train)
        a = accuracy_score(y_test, m.predict(X_test))
        print(f"  {k:<10} → {a * 100:.2f}%")

    # ── Save model & scaler ───────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n[INFO] Model  saved → {MODEL_PATH}")
    print(f"[INFO] Scaler saved → {SCALER_PATH}")


# ── 4. Predict ────────────────────────────────────────────────
def predict(data_path: str):
    # Load saved artefacts
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Model or scaler not found. Run training first:\n"
            "  python model.py --mode train --data heart.csv"
        )

    with open(MODEL_PATH, "rb") as f:
        model: SVC = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler: StandardScaler = pickle.load(f)

    print(f"[INFO] Loaded model  ← {MODEL_PATH}")
    print(f"[INFO] Loaded scaler ← {SCALER_PATH}")

    df = load_data(data_path)
    X_scaled, y_true, _ = preprocess(df, fit_scaler=False, scaler=scaler)

    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]  # probability of Disease

    results = df.copy()
    results["prediction"] = preds
    results["prediction_label"] = results["prediction"].map(
        {0: "No Disease", 1: "Disease"}
    )
    results["disease_probability"] = probs.round(3)

    print("\n[PREDICTIONS]")
    print(results[["prediction_label", "disease_probability"]].to_string())

    if y_true is not None:
        acc = accuracy_score(y_true, preds)
        print(f"\n[INFO] Accuracy on provided labels: {acc * 100:.2f}%")

    out_path = "predictions.csv"
    results.to_csv(out_path, index=False)
    print(f"[INFO] Predictions saved → {out_path}")


# ── 5. CLI entry point ────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Heart Disease SVM — train or predict"
    )
    parser.add_argument("--mode", choices=["train", "predict"], required=True,
                        help="'train' to fit model, 'predict' to run inference")
    parser.add_argument("--data", required=True,
                        help="Path to CSV file")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.data)
    else:
        predict(args.data)