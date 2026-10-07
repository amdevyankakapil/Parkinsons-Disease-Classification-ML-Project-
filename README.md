# Parkinson's Disease Classification Using Machine Learning

## 1. Project Overview

This is a simple first-year B.Tech AI-ML project that uses Machine Learning to classify a record as **Healthy** or **Parkinson's disease** based on biomedical voice measurements.

The project uses the **Oxford Parkinson's Disease Detection Dataset** from the UCI Machine Learning Repository.

The dataset contains voice measurements from people with and without Parkinson's disease. The `status` value is used as the target:

- `0` = Healthy
- `1` = Parkinson's disease

Source: UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/174/parkinsons

## 2. Technologies Used

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Logistic Regression

## 3. Machine Learning Steps

```text
Dataset
   ↓
Data checking
   ↓
Remove ID column
   ↓
Separate features and target
   ↓
Train-test split
   ↓
Feature scaling
   ↓
Logistic Regression
   ↓
Prediction
   ↓
Accuracy and confusion matrix
```

## 4. Main Features

Some important voice-related measurements in the dataset include:

- Fundamental frequency
- Jitter
- Shimmer
- HNR
- RPDE
- DFA
- PPE

These measurements are already extracted from voice recordings. The project does **not** record audio directly.

## 5. How to Run the Project

### Step 1: Install Python

Python 3.12 is recommended for this project.

### Step 2: Open a terminal in this project folder

### Step 3: Install the libraries

```bash
python -m pip install -r requirements.txt
```

If `python` does not work on Windows, try:

```bash
py -3.12 -m pip install -r requirements.txt
```

### Step 4: Run the program

```bash
python main.py
```

On Windows, you can also try:

```bash
py -3.12 main.py
```

The program automatically downloads the dataset through the `ucimlrepo` package.

## 6. Output

The program displays:

- Number of records
- Number of features
- Sample data
- Missing-value check
- Training and testing size
- Model accuracy
- Confusion matrix
- Precision
- Recall
- F1-score

It also saves a confusion matrix image inside the `results` folder.

## 7. Why Logistic Regression?

Logistic Regression is a beginner-friendly classification algorithm. It estimates the probability of a class and then assigns the observation to a class such as Healthy or Parkinson's.

## 8. Important Note

This project is made for educational purposes. It is **not a medical diagnostic system** and should not be used to diagnose Parkinson's disease.

## 9. Dataset Citation

Little, M. (2007). Parkinsons [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59C74.
