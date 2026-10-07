import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Load the same UCI dataset used by the main project
parkinsons = fetch_ucirepo(id=174)

X = parkinsons.data.features.copy()
y = parkinsons.data.targets.copy()

# Make target a simple Series
y = y.iloc[:, 0]

# Remove ID column
if "name" in X.columns:
    X = X.drop("name", axis=1)

# Remove duplicate columns
X = X.loc[:, ~X.columns.duplicated()]

# Same train/test split as the main project
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the same Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Predict test records
predictions = model.predict(X_test_scaled)

print("\n================================")
print("       MODEL CHECK")
print("================================\n")

for i in range(10):

    actual = y_test.iloc[i]
    predicted = predictions[i]

    actual_text = "Parkinson's" if actual == 1 else "Healthy"
    predicted_text = "Parkinson's" if predicted == 1 else "Healthy"

    print("Record", i + 1)
    print("Actual   :", actual_text)
    print("Predicted:", predicted_text)

    if actual == predicted:
        print("Result   : CORRECT")
    else:
        print("Result   : INCORRECT")

    print("----------------------------")