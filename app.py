import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------------
# 1. Project configuration
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "churn_prediction_model.joblib"

# -----------------------------------
# 2. Page configuration
# -----------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------
# 3. Load trained model
# -----------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except FileNotFoundError:
    st.error(
        "Trained model not found. "
        "Please run the Jupyter Notebook first."
    )
    st.stop()


# -----------------------------------
# 4. Dashboard heading
# -----------------------------------

st.title("📊 Customer Churn Prediction System")

st.write(
    "Predict whether a telecom customer "
    "is likely to leave the company."
)

st.divider()


# -----------------------------------
# 5. Dashboard information
# -----------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("ML Algorithm", "Random Forest")

with col2:
    st.metric("Application", "Churn Prediction")

with col3:
    st.metric("Dataset", "IBM Telco")


st.divider()


# -----------------------------------
# 6. Customer information form
# -----------------------------------

st.header("Enter Customer Information")

with st.form("customer_form"):

    st.subheader("Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Has Partner?",
            ["Yes", "No"]
        )

    with col2:
        dependents = st.selectbox(
            "Has Dependents?",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Customer Tenure (Months)",
            0, 72, 12
        )

    st.divider()

    st.subheader("Service Information")

    col1, col2 = st.columns(2)

    with col1:
        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    st.divider()

    st.subheader("Contract and Billing Information")

    col1, col2 = st.columns(2)

    with col1:
        contract = st.selectbox(
            "Contract Type",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col2:
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=100000.0,
            value=840.0,
            step=1.0
        )

    st.divider()

    predict_button = st.form_submit_button(
        "🔍 Predict Customer Churn",
        use_container_width=True
    )


# -----------------------------------
# 7. Make prediction
# -----------------------------------

if predict_button:

    customer_data = pd.DataFrame({

        "gender": [gender],

        "SeniorCitizen": [senior_citizen],

        "Partner": [partner],

        "Dependents": [dependents],

        "tenure": [tenure],

        "PhoneService": [phone_service],

        "MultipleLines": [multiple_lines],

        "InternetService": [internet_service],

        "OnlineSecurity": [online_security],

        "OnlineBackup": [online_backup],

        "DeviceProtection": [device_protection],

        "TechSupport": [tech_support],

        "StreamingTV": [streaming_tv],

        "StreamingMovies": [streaming_movies],

        "Contract": [contract],

        "PaperlessBilling": [paperless_billing],

        "PaymentMethod": [payment_method],

        "MonthlyCharges": [monthly_charges],

        "TotalCharges": [total_charges]

    })

    prediction = model.predict(customer_data)[0]

    probability = model.predict_proba(
        customer_data
    )[0][1]

    # -----------------------------------
    # 8. Display results
    # -----------------------------------

    st.divider()

    st.header("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Prediction",
            "Likely to Leave" if prediction == 1 else "Likely to Stay"
        )

    if prediction == 1:

        st.error(
            "⚠️ The model predicts that this "
            "customer may leave the company."
        )

        st.write(
            "The company may consider contacting "
            "the customer and reviewing their needs."
        )

    else:

        st.success(
            "✅ The model predicts that this "
            "customer is likely to stay."
        )

    st.subheader("Customer Information")

    st.dataframe(
        customer_data,
        use_container_width=True
    )

    st.caption(
        "Predictions are estimates based on historical "
        "data and are not guaranteed outcomes."
    )