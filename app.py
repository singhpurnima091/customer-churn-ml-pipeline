import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/churn_pipeline.pkl")

st.title("📊 Customer Churn Prediction App")

# -------------------------------
# Inputs
# -------------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
tenure = st.slider("Tenure Months", 0, 72)

phone = st.selectbox("Phone Service", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

payment = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

monthly = st.number_input("Monthly Charges", 0.0, 200.0)
total = st.number_input("Total Charges", 0.0, 10000.0)

# -------------------------------
# Predict
# -------------------------------
if st.button("Predict"):

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Senior Citizen": senior,
        "Partner": partner,
        "Dependents": "No",
        "Tenure Months": float(tenure),
        "Phone Service": phone,
        "Multiple Lines": "No",
        "Internet Service": internet,
        "Online Security": "No",
        "Online Backup": "Yes",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": contract,
        "Paperless Billing": "Yes",
        "Payment Method": payment,
        "Monthly Charges": float(monthly),
        "Total Charges": float(total)
    }])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer will churn (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer will NOT churn (Probability: {prob:.2f})")