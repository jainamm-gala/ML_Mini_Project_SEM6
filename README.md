
# Loan Approval Prediction

This project aims to predict loan approval status based on various applicant features using machine learning models. The goal is to develop a robust model that can assist financial institutions in making informed decisions regarding loan applications.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Feature Importance](#feature-importance)
- [Prediction Demo](#prediction-demo)
- [Interactive Predictor](#interactive-predictor)

## Project Overview

This repository contains a Colab notebook that demonstrates the process of building and evaluating machine learning models for loan approval prediction. Three different classification algorithms—Logistic Regression, Decision Tree, and Support Vector Machine (SVM)—are implemented and compared. An interactive tool for predicting loan status for new applicants is also included.

## Dataset

The dataset `loan_approval_dataset.csv` contains various features related to loan applicants, including:
- `no_of_dependents`: Number of dependents
- `education`: Applicant's education level
- `self_employed`: Whether the applicant is self-employed
- `income_annum`: Annual income of the applicant
- `loan_amount`: Requested loan amount
- `loan_term`: Loan term in years
- `cibil_score`: CIBIL (credit) score of the applicant
- `residential_assets_value`: Value of residential assets
- `commercial_assets_value`: Value of commercial assets
- `luxury_assets_value`: Value of luxury assets
- `bank_asset_value`: Value of bank assets
- `loan_status`: Loan approval status (Target variable: 1 for Approved, 0 for Rejected)

## Data Preprocessing

The initial steps involved cleaning the data and preparing it for model training:

1.  **Drop unnecessary column**: The `loan_id` column was dropped as it's not relevant for prediction.
2.  **Strip spaces from column names**: Column names were cleaned to remove leading/trailing spaces for easier access.
3.  **Encoding Categorical Features**: 
    - `education`: 'Graduate' mapped to 1, 'Not Graduate' to 0.
    - `self_employed`: 'Yes' mapped to 1, 'No' to 0.
    - `loan_status`: 'Approved' mapped to 1, 'Rejected' to 0.
4.  **Missing Values**: Checked for missing values. The dataset was found to have no missing values, so no imputation was required.

## Exploratory Data Analysis (EDA)

### Correlation Matrix

A correlation matrix was generated to visualize the relationships between numerical features.

![Correlation Matrix](https://via.placeholder.com/600x400/CCCCCC/FFFFFF?text=Correlation+Matrix)
*Replace with actual image link of correlation matrix plot*

### CIBIL Score Distribution by Loan Status

A box plot was used to visualize the distribution of CIBIL scores for approved vs. rejected loans.

![CIBIL Score Distribution](https://via.placeholder.com/600x400/CCCCCC/FFFFFF?text=CIBIL+Score+Distribution)
*Replace with actual image link of CIBIL Score Distribution plot*

## Model Training

The dataset was split into training and testing sets (80% train, 20% test). The following features were selected for training:
- `cibil_score`
- `loan_term`
- `loan_amount`
- `income_annum`

Three classification models were trained on the preprocessed data:

1.  **Logistic Regression**
2.  **Decision Tree Classifier**
3.  **Support Vector Machine (SVM)**

## Model Evaluation

Each model was evaluated using standard metrics: Accuracy, Confusion Matrix, and Classification Report.

### Logistic Regression
```
--- Logistic Regression ---
Accuracy: 0.7927400468384075
Confusion Matrix:
 [[ 50 258]
 [ -2 548]]
Classification Report:
               precision    recall  f1-score   support

           0       0.96      0.16      0.28       308
           1       0.68      1.00      0.81       546

    accuracy                           0.79       854
   macro avg       0.82      0.58      0.55       854
weighted avg       0.78      0.79      0.71       854

```

### Decision Tree
```
--- Decision Tree ---
Accuracy: 0.9800936768149883
Confusion Matrix:
 [[305   3]
 [ 14 532]]
Classification Report:
               precision    recall  f1-score   support

           0       0.96      0.99      0.97       308
           1       0.99      0.97      0.98       546

    accuracy                           0.98       854
   macro avg       0.98      0.98      0.98       854
weighted avg       0.98      0.98      0.98       854

```

### SVM
```
--- SVM ---
Accuracy: 0.6276346604215457
Confusion Matrix:
 [[  0 308]
 [  0 546]]
Classification Report:
               precision    recall  f1-score   support

           0       0.00      0.00      0.00       308
           1       0.64      1.00      0.78       546

    accuracy                           0.63       854
   macro avg       0.32      0.50      0.39       854
weighted avg       0.40      0.63      0.49       854

```

### Model Accuracy Comparison

![Model Accuracy Comparison](https://via.placeholder.com/600x400/CCCCCC/FFFFFF?text=Model+Accuracy+Comparison)
*Replace with actual image link of model accuracy comparison plot*

The Decision Tree model achieved the highest accuracy of approximately 98%, outperforming Logistic Regression (79%) and SVM (63%). The SVM model's performance was notably poor, classifying almost all loans as approved.

## Feature Importance

Feature importance was calculated using the Decision Tree model to identify the most influential factors in loan approval.

![Feature Importance](https://via.placeholder.com/600x400/CCCCCC/FFFFFF?text=Feature+Importance)
*Replace with actual image link of feature importance plot*

`cibil_score` was identified as the most important feature, followed by `loan_term`, `loan_amount`, and `income_annum`.

## Prediction Demo

A simple example demonstrates how to predict the loan approval status for a new applicant using the Decision Tree model.

**Sample Applicant Data:**
```
 cibil_score  loan_term  loan_amount  income_annum
         750         15     20000000       7000000
```

**Prediction Output:**
```
Probability of Loan Rejection: 0.00%
Probability of Loan Approval: 100.00%
Based on the model, the loan is Approved.
```

## Interactive Predictor

The notebook also includes an interactive widget-based predictor where you can adjust parameters like CIBIL score, loan term, loan amount, and annual income to see real-time loan approval predictions. This uses the Decision Tree model.

*Note: The interactive widgets will not function directly in the static GitHub preview. You need to run the notebook in a Colab environment to interact with them.*
