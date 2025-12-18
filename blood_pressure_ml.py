# Blood Pressure Abnormality Prediction
# ------------------------------------
# This script trains two models (Logistic Regression & Random Forest)
# to predict whether a patient has abnormal blood pressure.

import os

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Download dataset from Kaggle
path = kagglehub.dataset_download("pavanbodanki/blood-press")
print("Path to dataset files:", path)
print("Files in dataset directory:")
print(os.listdir(path))


# Load dataset
df = pd.read_csv(os.path.join(path, "data.csv"))

# Quick look at the data
print("\nDataset Preview:")
print(df.head())


# Remove rows with missing values
df = df.dropna()


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Standardize features (important for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

# Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Predictions and accuracy
print("\nModel Accuracies:")
print("Logistic Regression Accuracy:", lr_accuracy)
print("Random Forest Accuracy:", rf_accuracy)

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Normal", "Abnormal"]
)
disp.plot()
plt.title("Blood Pressure Abnormality Classification")
plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.show()

# Feature Importance Plot
importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance for BP Abnormality Prediction")
plt.savefig("feature_importance.png", bbox_inches="tight")
plt.show()

# Model Comparison Plot
models = ['Logistic Regression', 'Random Forest']
accuracies = [lr_accuracy, rf_accuracy]

plt.figure(figsize=(6, 4))
plt.bar(models, accuracies)
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.ylim(0, 1)
plt.savefig("model_comparison.png", bbox_inches="tight")
plt.show()

