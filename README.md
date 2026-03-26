## 🌐 Live Demo
https://customer-churn-ml-pipeline-epgfkj8sakrjaat5dvvbvq.streamlit.app/

## 🚀 Features
- End-to-end ML pipeline
- Streamlit web app
- FastAPI backend
- Real-time prediction

## 🧠 Tech Stack
- Python
- Scikit-learn
- Streamlit
- FastAPI
# Customer Churn Prediction ML Pipeline

This project builds a Machine Learning pipeline to predict customer churn using telecom customer data.

The pipeline performs data preprocessing, feature encoding, model training, and prediction using Python and Scikit-learn.

## Project Structure

customer-churn-ml-pipeline/
│
├── data/
│   ├── raw.csv
│   ├── processed.csv
│   └── Telco_customer_churn.xlsx
│
├── models/
│   └── churn_model.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict.py
│
├── requirements.txt
└── README.md

## Workflow

1. Load telecom churn dataset
2. Clean and preprocess data
3. Encode categorical features
4. Split data into train and test sets
5. Train Logistic Regression model
6. Evaluate model performance
7. Save trained model
8. Make predictions

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NumPy
- Joblib

## How to Run the Project

### 1. Clone the repository
git clone https://github.com/singhpurnima091/customer-churn-ml-pipeline.git
cd customer-churn-ml-pipeline
### 2. Create virtual environment

python -m venv venv
venv\Scripts\activate
### 3. Install dependencies

pip install -r requirements.txt
### 4. Run preprocessing

python src/preprocess.py
### 5. Train model

python src/train_model.py
### 6. Run prediction

python src/predict.py
## Model Used

Logistic Regression classifier for binary churn prediction.

## Output

- Model saved in models/churn_model.pkl
- Predictions generated from processed dataset.

## Future Improvements

- Hyperparameter tuning
- Feature scaling
- Model comparison (Random Forest, XGBoost)
- Deploy model using FastAPI
