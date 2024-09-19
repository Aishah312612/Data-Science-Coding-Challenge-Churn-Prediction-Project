# Churn Prediction Project

This project focuses on predicting customer churn based on user data, using machine learning techniques such as logistic regression. The primary objective is to classify whether a customer will churn (leave the service) or stay, based on features like account age, monthly charges, device preferences, and more.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)

## Overview

Customer churn is a critical issue for many businesses. By predicting churn, companies can take action to retain customers. This project uses logistic regression to predict customer churn probabilities based on a variety of customer features.

## Dataset

The dataset consists of customer data, including features such as:

- **Account Age**: How long the customer has had the account.
- **Monthly Charges**: The amount the customer pays monthly.
- **Genre Preference**: The customer's preferred content genre (e.g., Comedy, Drama, Sci-Fi).
- **Device Preferences**: Whether the customer uses multiple devices, has a registered device, etc.

The dataset has been split into training and test sets:
- `train.csv`: Contains the training data, including the target variable (`Churn`).
- `test.csv`: Contains the test data, excluding the target variable (`Churn`).

## Feature Engineering

To prepare the data for machine learning, the following steps were performed:

1. **Categorical Encoding**: 
   - Columns such as `GenrePreference`, `SubscriptionType`, and `Gender` were one-hot encoded using `pd.get_dummies()` to convert categorical variables into numerical form.
   
2. **Feature Standardization**: 
   - The continuous variables like `MonthlyCharges` and `AccountAge` were standardized using `StandardScaler` to ensure they are on the same scale.
   
3. **Missing Value Handling**: 
   - Missing values in test data were handled by creating any necessary columns and filling missing entries with 0, ensuring alignment with the training data structure.

## Model Training

The Logistic Regression model was trained on the processed training data using the following steps:

1. **Data Preparation**: 
   - Features and target labels were separated. Categorical features were one-hot encoded and standardized.
   
2. **Logistic Regression**: 
   - The model was trained using logistic regression, which is suitable for binary classification tasks like churn prediction.

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

## Prediction

The trained model was used to predict the probability of churn on the test data:

1. **Preprocessing Test Data**: 
   - The same preprocessing steps applied to the training data were applied to the test data, including encoding and scaling.
   
2. **Predicting Probabilities**: 
   - The logistic regression model was used to predict churn probabilities on the test dataset.

```python
predicted_probabilities = model.predict_proba(X_test)[:, 1]
```

3. **Creating the Submission File**: 
   - A submission file containing customer IDs and their predicted churn probabilities was generated.

```python
prediction_df = pd.DataFrame({
    'CustomerID': test_df['CustomerID'],
    'predicted_probability': predicted_probabilities
})
```

## Results

The model outputs a probability score between 0 and 1, indicating the likelihood of a customer churning. Customers with a higher probability (e.g., above 0.5) are predicted to churn.

## Usage

To use this project:

1. **Prepare the data** by ensuring the `train.csv` and `test.csv` files are available.
2. **Run the feature engineering and model training** to fit the logistic regression model on the training data.
3. **Use the trained model** to predict churn probabilities for the test data and generate the submission file.

## Contributing

Contributions are welcome! Feel free to submit issues, improvements, or pull requests.
