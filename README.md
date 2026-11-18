# Heart Disease Classification — Support Vector Machine

A supervised-learning project that applies Support Vector Machines (SVM) to a 13-feature heart-disease dataset.

> **Medical disclaimer:** This is an educational ML project, not a clinical diagnostic system. Do not use its outputs for medical decisions.

## Dataset
The project uses a Kaggle heart-disease dataset with 13 input features and a binary target. The README documents clinical and exercise-test variables such as age, resting blood pressure, cholesterol, maximum heart rate, and ST depression.

Dataset source: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Download `heart.csv` into the project root.

## Model workflow
- Split data into training and test sets (80/20 as documented).
- Standardize features using StandardScaler.
- Train an SVM with an RBF kernel.
- Compare linear, RBF, polynomial, and sigmoid kernels.
- Generate a classification report and confusion matrix.
- Save the fitted model and scaler for inference.

## Run
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
python model.py --mode train --data heart.csv
python model.py --mode predict --data heart.csv
```

## Stack
Python · Pandas · NumPy · scikit-learn · Matplotlib · Seaborn

## Evaluation note
The README contains typical kernel accuracy estimates, not a guarantee of repeatable performance. Re-run the pipeline with a fixed seed and report the actual held-out metrics before comparing models.

## Author
[Aun Abbas](https://github.com/aunabbas007)
