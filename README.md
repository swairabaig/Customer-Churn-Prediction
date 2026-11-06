# Customer Churn Prediction System

## Project Overview

The Customer Churn Prediction System is a Machine Learning project that predicts whether a telecom customer is likely to leave the company or continue using its services.

The project uses a **Random Forest Classifier** to analyze customer information, identify important factors related to customer churn, and make predictions.

The main goal is to help telecom companies identify customers who may leave so they can take steps to improve customer retention.

## Objectives

- Analyze customer data to understand churn behavior.
- Clean and preprocess the dataset.
- Perform Exploratory Data Analysis (EDA).
- Train a Random Forest classification model.
- Evaluate model performance using different metrics.
- Identify the most important features affecting customer churn.
- Save the trained model for future predictions.
- Develop a Streamlit application for customer churn prediction.

## Dataset

**Dataset:** IBM Telco Customer Churn Dataset

**Source:** https://github.com/IBM/telco-customer-churn-on-icp4d

Dataset information:

- Total records: 7,043
- Total original columns: 21
- Input features after removing customerID: 19
- Target variable: Churn

The target variable contains two classes:

- 0 = Customer Stayed
- 1 = Customer Left

### Dataset Distribution

| Customer Category | Number | Percentage |
|---|---:|---:|
| Stayed | 5,174 | 73.46% |
| Left | 1,869 | 26.54% |
| Total | 7,043 | 100% |

## Technologies and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Project Workflow

### 1. Data Collection
Downloaded the IBM Telco Customer Churn dataset.

### 2. Data Cleaning
- Checked missing values.
- Checked duplicate records.
- Removed duplicate rows.
- Converted TotalCharges into numeric format.
- Removed the customerID column.
- Handled missing values.

### 3. Exploratory Data Analysis (EDA)

Created visualizations to understand:

- Customer churn distribution.
- Customer churn by contract type.
- Customer tenure and churn.
- Monthly charges and churn.

### 4. Data Preprocessing

- Separated input features and target variable.
- Identified numerical and categorical features.
- Used median imputation for numerical features.
- Used most-frequent imputation for categorical features.
- Applied One-Hot Encoding to categorical features.
- Split the data into training and testing sets.

### 5. Model Training

Used the Random Forest Classifier with the following parameters:

| Parameter | Value |
|---|---|
| n_estimators | 200 |
| max_depth | 15 |
| min_samples_split | 5 |
| min_samples_leaf | 2 |
| class_weight | balanced |
| random_state | 42 |

Training data: 80%

Testing data: 20%

### 6. Model Evaluation

The trained model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Classification Report
- Confusion Matrix
- ROC Curve

## Model Results

The Random Forest model achieved the following results on the test dataset:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 75.16% |
| Precision | 52.37% |
| Recall | 70.86% |
| F1-score | 60.23% |
| ROC-AUC | 83.43% |

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Stayed | 0.88 | 0.77 | 0.82 |
| Left | 0.52 | 0.71 | 0.60 |

The model achieved 83.43% ROC-AUC, indicating its ability to distinguish between customers who stay and customers who leave.

## Important Features

According to the Random Forest feature importance analysis, the top factors influencing predictions include:

1. Tenure
2. Total Charges
3. Month-to-month Contract
4. Monthly Charges
5. Two-year Contract
6. Online Security
7. Tech Support
8. Internet Service
9. Payment Method
10. Online Backup

These features were identified using the trained model's feature importance values.

## Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   ├── customer_churn.csv
│   └── test_predictions.csv
│
├── images/
│   ├── churn_distribution.png
│   ├── contract_churn.png
│   ├── tenure_churn.png
│   ├── monthly_charges_churn.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── models/
│   └── churn_prediction_model.joblib
│
├── Customer_Churn_Analysis.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### 2. Open the Project Folder

```bash
cd Customer-Churn-Prediction
```

### 3. Install Required Libraries

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook

Open `Customer_Churn_Analysis.ipynb` in Jupyter Notebook or VS Code to explore the complete machine learning workflow.

### 5. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to use the application.

## Saved Model

The trained Random Forest model is saved at:

`models/churn_prediction_model.joblib`

The model can be loaded and reused to predict customer churn without training it again.

## Future Improvements

- Improve model accuracy and precision.
- Experiment with other classification algorithms.
- Apply hyperparameter tuning.
- Improve the Streamlit dashboard.
- Add more customer insights and visualizations.
- Deploy the application online.

## Conclusion

This project demonstrates an end-to-end Machine Learning workflow for customer churn prediction.

It includes data collection, data cleaning, exploratory data analysis, preprocessing, Random Forest model training, evaluation, feature importance analysis, model saving, and a Streamlit application.

The system can help telecom companies identify customers who may leave and support better customer retention decisions.

---

**Project:** Customer Churn Prediction System  
**Algorithm:** Random Forest Classifier  
**Language:** Python  
**Interface:** Streamlit
