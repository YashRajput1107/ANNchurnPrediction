Customer Churn Prediction using ANN

An end-to-end deep learning project that predicts whether a bank customer is likely to churn (leave the bank) using an Artificial Neural Network (ANN) built with TensorFlow/Keras.

📋 Overview

This project uses the Churn Modelling dataset to train a binary classification model that predicts customer churn based on demographic and account information. The pipeline covers data preprocessing, model training, evaluation, and inference on new/unseen data.

🗂️ Project Structure

.
├── Churn_Modelling.csv          # Raw dataset
├── experiments.ipynb            # Data preprocessing, EDA, model training & evaluation
├── prediction.ipynb             # Loads saved model/encoders and runs predictions on new data
├── model.h5                     # Trained ANN model (saved weights + architecture)
├── label_encoder_gender.pkl     # Fitted LabelEncoder for the 'Gender' column
├── onehot_encoder_geo.pkl       # Fitted OneHotEncoder for the 'Geography' column
├── scaler.pkl                   # Fitted StandardScaler for numerical features
├── logs/                        # TensorBoard training logs
├── requirements.txt             # Python dependencies
└── README.md

📊 Dataset

The dataset (Churn_Modelling.csv) contains 10,000 customer records with the following features:

FeatureDescriptionCreditScoreCustomer's credit scoreGeographyCustomer's country (France, Germany, Spain)GenderMale / FemaleAgeCustomer's ageTenureNumber of years as a bank customerBalanceAccount balanceNumOfProductsNumber of bank products usedHasCrCardWhether the customer has a credit card (1/0)IsActiveMemberWhether the customer is an active member (1/0)EstimatedSalaryEstimated annual salaryExitedTarget variable — 1 if the customer churned, 0 otherwise

⚙️ Preprocessing Steps


Dropped irrelevant columns (RowNumber, CustomerId, Surname).
Encoded Gender using LabelEncoder.
Encoded Geography using OneHotEncoder (since it has more than two categories).
Scaled numerical features using StandardScaler.
Split the data into training and test sets.
Saved all fitted encoders/scaler as .pkl files for reuse during inference.
