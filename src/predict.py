import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

print("Loading model...")
model = joblib.load("models/churn_model.pkl")

print("Loading dataset...")
df = pd.read_csv("data/processed.csv")

# Drop columns same as training
drop_cols = [
    "Country","State","City","Zip Code","Lat Long","Latitude","Longitude",
    "Churn Score","CLTV","Churn Reason","Churn Label"
]

for col in drop_cols:
    if col in df.columns:
        df = df.drop(col, axis=1)

# Encode categorical columns
le = LabelEncoder()

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = le.fit_transform(df[column])

# Separate features
X = df.drop("Churn Value", axis=1)

# Select one sample
sample = X.iloc[[0]]

print("Making prediction...")

prediction = model.predict(sample)
prob = model.predict_proba(sample)

print("\nPrediction Result")

if prediction[0] == 1:
    print("Customer will churn")
else:
    print("Customer will NOT churn")

print("Probability:", prob.max())