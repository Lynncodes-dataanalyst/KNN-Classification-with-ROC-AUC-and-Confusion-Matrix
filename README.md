Project Overview

This project applies a K-Nearest Neighbors (KNN) classifier to a dataset, evaluates its performance with ROC-AUC curves and a confusion matrix, and explores how dataset imbalance affects results.

Originally, the model gave an AUC of 0.500 because predicted class labels (instead of probabilities) were used. This was fixed by using predict_proba and stratified splitting to ensure all classes are represented.

Features

Data preprocessing with PCA (2D compressed features)

Stratified train-test split

Training a KNeighborsClassifier

Multiclass ROC curve and AUC calculation

Confusion matrix visualization

Classification report (precision, recall, F1-score)

Dataset

843 total samples

3 target classes, highly imbalanced:

Class 0 → 32 samples

Class 1 → 759 samples

Class 2 → 42 samples

Installation

Clone this repo and install dependencies:

git clone https://github.com/Lynncodes-dataanalyst/KNN-Classification-with-ROC-AUC-and-Confusion-Matrix
cd your-repo
pip install -r requirements.txt


Requirements (key packages):

numpy
pandas
scikit-learn
matplotlib
seaborn

Usage

Run the notebook or Python script to train and evaluate the model.

Key steps in the code:

# Train
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict probabilities
y_proba = knn.predict_proba(X_test)

# ROC-AUC
overall_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

# Confusion Matrix
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

Results

Accuracy: ~91%

ROC-AUC: >0.5 after using probabilities

Confusion Matrix: dominated by the majority class, but minor classes are now recognized

Classification Report: gives per-class precision, recall, and F1

Next Steps

Handle class imbalance (e.g., SMOTE oversampling)

Try other classifiers (SVM, Random Forest)

Hyperparameter tuning (n_neighbors, distance metrics)
