"""
Parkinson's Disease Classification Using Machine Learning
First-year B.Tech AI-ML project

This program downloads the official UCI Parkinson's dataset,
prepares the data, trains a Logistic Regression model, and
checks the model using common classification measures.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -----------------------------
# 1. Load the dataset
# -----------------------------
print("Loading Parkinson's dataset...")

parkinsons = fetch_ucirepo(id=174)
X = parkinsons.data.features.copy()
y = parkinsons.data.targets.copy()

# Make sure the target is a simple Series.
y = y.iloc[:, 0]

print("Dataset loaded successfully.")
print("Number of records:", len(X))
print("Number of features:", X.shape[1])

# -----------------------------
# 2. Basic data checking
# -----------------------------
print("\nFirst five rows:")
print(X.head())

print("\nMissing values:")
print(X.isnull().sum().sum())

# The 'name' column is an ID, not a useful voice measurement.
if "name" in X.columns:
    X = X.drop("name", axis=1)
X = X.loc[:, ~X.columns.duplicated()]

# -----------------------------
# 3. Split the data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# -----------------------------
# 4. Scale the features
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Train Logistic Regression
# -----------------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print("\nModel training completed.")

# -----------------------------
# 6. Make predictions
# -----------------------------
y_pred = model.predict(X_test_scaled)

# -----------------------------
# 7. Evaluate the model
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Healthy", "Parkinson's"]
))

# -----------------------------
# 8. Save confusion matrix plot
# -----------------------------
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Healthy", "Parkinson's"])
plt.yticks([0, 1], ["Healthy", "Parkinson's"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

print("\nConfusion matrix saved in results/confusion_matrix.png")
print("\nProject completed successfully!")
print("Note: This is an educational ML project and is not a medical diagnosis tool.")
