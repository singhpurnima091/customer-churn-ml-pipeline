from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# Load model
model = joblib.load("models/churn_pipeline.pkl")

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: dict):

    input_data = pd.DataFrame([{
        "Gender": data["Gender"],
        "Senior Citizen": data["Senior Citizen"],
        "Partner": data["Partner"],
        "Dependents": "No",
        "Tenure Months": float(data["Tenure Months"]),
        "Phone Service": data["Phone Service"],
        "Multiple Lines": "No",
        "Internet Service": data["Internet Service"],
        "Online Security": "No",
        "Online Backup": "Yes",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": data["Contract"],
        "Paperless Billing": "Yes",
        "Payment Method": data["Payment Method"],
        "Monthly Charges": float(data["Monthly Charges"]),
        "Total Charges": float(data["Total Charges"])
    }])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(prob)
    }