# Heart Disease Prediction — SVM

A beginner-friendly Support Vector Machine (SVM) project that predicts the presence of heart disease from 13 clinical features. Part of a step-by-step ML learning series.

---

## Project Structure

```
heart-disease-svm/
├── model.py                  ← main script (train + predict)
├── heart.csv                 ← dataset (download from Kaggle)
├── svm_heart_model.pkl       ← saved model (generated after training)
├── svm_heart_scaler.pkl      ← saved scaler (generated after training)
├── confusion_matrix.png      ← evaluation plot (generated after training)
├── predictions.csv           ← prediction output (generated after predict)
└── README.md
```

---

## Dataset

**Heart Disease Dataset** from Kaggle:  
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Download `heart.csv` and place it in the project root.

### Features

| Feature    | Description                          | Type        |
|------------|--------------------------------------|-------------|
| age        | Age in years                         | Numeric     |
| sex        | 1 = male, 0 = female                 | Categorical |
| cp         | Chest pain type (0–3)                | Categorical |
| trestbps   | Resting blood pressure (mm Hg)       | Numeric     |
| chol       | Serum cholesterol (mg/dl)            | Numeric     |
| fbs        | Fasting blood sugar > 120 mg/dl      | Categorical |
| restecg    | Resting ECG results (0–2)            | Categorical |
| thalach    | Max heart rate achieved              | Numeric     |
| exang      | Exercise-induced angina (1=yes)      | Categorical |
| oldpeak    | ST depression induced by exercise    | Numeric     |
| slope      | Slope of peak exercise ST segment    | Categorical |
| ca         | Number of major vessels (0–3)        | Numeric     |
| thal       | Thalassemia type (0–3)               | Categorical |
| **target** | **0 = No disease, 1 = Disease**      | **Target**  |

---

## Setup

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## Usage

### Train the model

```bash
python model.py --mode train --data heart.csv
```

What this does:
- Loads and validates the dataset
- Splits into 80% train / 20% test
- Scales features using `StandardScaler` (required for SVM)
- Trains an SVM with `kernel='rbf'`
- Prints accuracy + classification report
- Saves confusion matrix plot → `confusion_matrix.png`
- Compares all 4 kernels (linear, rbf, poly, sigmoid)
- Saves model → `svm_heart_model.pkl`
- Saves scaler → `svm_heart_scaler.pkl`

### Run predictions

```bash
python model.py --mode predict --data heart.csv
```

What this does:
- Loads the saved `.pkl` model and scaler
- Runs inference on your CSV
- Outputs prediction label + disease probability per row
- Saves results → `predictions.csv`

---

## How SVM Works (Quick Summary)

SVM finds the **hyperplane** (decision boundary) that **maximizes the margin** between the two classes. The data points closest to the boundary are called **support vectors** — they are the only points that define the boundary.

```
Class A  |  margin  |  Class B
  ●  ●   |──────────|   ■  ■
  ●  ●*  |          |  *■  ■
         ^          ^
    margin edge   margin edge
         (support vectors marked with *)
```

### Why feature scaling is mandatory

SVM computes distances between points. Without scaling, a feature like `chol` (range 0–300) would dominate `sex` (range 0–1). `StandardScaler` brings every feature to mean=0, std=1.

### The kernel trick

When data is not linearly separable, the `rbf` kernel implicitly maps data into a higher-dimensional space where a separating hyperplane exists — without ever computing that transformation explicitly.

### Key hyperparameters

| Parameter | Effect |
|-----------|--------|
| `C` | Regularization. High C = smaller margin, fits train data harder. Low C = wider margin, more robust. |
| `kernel` | `rbf` is the default and works well for most problems. |
| `gamma` | Controls how far the influence of a single training example reaches. `'scale'` is a safe default. |

---

## Results (typical on this dataset)

| Kernel  | Accuracy |
|---------|----------|
| linear  | ~85%     |
| rbf     | ~87%     |
| poly    | ~82%     |
| sigmoid | ~75%     |

> Results may vary slightly based on random seed and train/test split.
