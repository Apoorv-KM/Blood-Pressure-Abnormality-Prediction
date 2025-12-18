# Blood Pressure Abnormality Prediction

This project predicts whether a patient has abnormal blood pressure using clinical
and lifestyle features. The dataset is sourced from Kaggle.

## Models Used
- Logistic Regression
- Random Forest Classifier

## Results

### Confusion Matrix (Random Forest)
![Confusion Matrix](confusion_matrix.png)

### Feature Importance (Random Forest)
![Feature Importance](feature_importance.png)

### Model Accuracy Comparison
![Model Comparison](model_comparison.png)

## Accuracy Summary
- Logistic Regression Accuracy: 0.75
- Random Forest Accuracy: 0.875

## Dataset
Kaggle – Blood Pressure Dataset


```bash
pip install pandas numpy matplotlib scikit-learn kagglehub
python blood_pressure_ml.py

Save it.

---

## 🚀 STEP 2: Push to GitHub (Clean Way)

Open terminal in that folder and run:

```bash
git init
git add .
git commit -m "Initial commit: BP abnormality prediction using ML"
git branch -M main
git remote add origin https://github.com/Apoorv-KM/Blood-Pressure-Abnormality-Prediction.git
git push -u origin main
