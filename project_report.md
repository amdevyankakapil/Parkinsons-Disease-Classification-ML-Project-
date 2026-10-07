# Parkinson's Disease Classification Using Machine Learning

## 1. Introduction

Parkinson's disease is a neurological disorder that can affect movement and speech. Changes in a person's voice can be measured using different biomedical voice features.

In this project, Machine Learning is used to classify records into two classes: Healthy and Parkinson's disease.

## 2. Problem Statement

To build a Machine Learning classification model that can classify a person's voice-measurement record as Healthy or Parkinson's disease using the UCI Parkinson's dataset.

## 3. Objective

The main objectives are:

1. Understand a real-world healthcare dataset.
2. Prepare the dataset for Machine Learning.
3. Split the data into training and testing sets.
4. Train a Logistic Regression classification model.
5. Make predictions on unseen test data.
6. Evaluate the model using accuracy, precision, recall, F1-score and a confusion matrix.

## 4. Dataset

The project uses the Oxford Parkinson's Disease Detection Dataset from the UCI Machine Learning Repository.

The dataset contains biomedical voice measurements from 31 people and uses the `status` column as the target. Status 0 represents healthy records and status 1 represents Parkinson's disease records.

The dataset has no missing values according to UCI.

## 5. Features

Examples of features include:

- Fundamental frequency: measures related to voice pitch.
- Jitter: variation in fundamental frequency.
- Shimmer: variation in voice amplitude.
- HNR: ratio related to harmonic and noise components.
- RPDE: a nonlinear voice-related measure.
- DFA: a signal scaling measure.
- PPE: a measure related to fundamental-frequency variation.

## 6. Methodology

### Step 1: Data Collection

The official UCI Parkinson's dataset is loaded using the `ucimlrepo` Python package.

### Step 2: Data Preparation

The `name` column is removed because it identifies a recording and is not a useful voice measurement for the model.

The remaining numerical columns are used as input features.

### Step 3: Train-Test Split

The data is divided into 80% training data and 20% testing data.

The training data is used to teach the model, while the testing data is used to check the model on records it did not see during training.

### Step 4: Feature Scaling

StandardScaler is used to put the numerical features on a similar scale. The scaler is fitted only on the training data and then applied to the test data.

### Step 5: Model Training

Logistic Regression is used because the target has two classes.

### Step 6: Prediction

The trained model predicts the class of the test records.

### Step 7: Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## 7. Hypothesis

**H1:** Voice-related biomedical measurements contain useful patterns that can help a Machine Learning classification model distinguish between Healthy and Parkinson's disease records.

**H0:** Voice-related biomedical measurements do not provide useful patterns for distinguishing between Healthy and Parkinson's disease records using the selected Machine Learning model.

## 8. Expected Result

The Logistic Regression model should learn patterns from the training data and produce predictions for the unseen test data. The final performance will be measured using the evaluation metrics printed by the program.

## 9. Conclusion

This project demonstrates a basic Machine Learning classification workflow using a real-world healthcare dataset. It covers data loading, data preparation, train-test splitting, scaling, model training, prediction and evaluation.

The project is educational and should not be treated as a medical diagnostic tool.
