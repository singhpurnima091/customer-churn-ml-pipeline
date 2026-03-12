import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

print("Loading processed dataset...")

df = pd.read_csv("data/processed.csv")

# Drop columns not needed for prediction
drop_cols = [
    "Country","State","City","Zip Code","Lat Long","Latitude","Longitude",
    "Churn Score","CLTV","Churn Reason","Churn Label"
]

for col in drop_cols:
    if col in df.columns:
        df = df.drop(col, axis=1)

# Encode categorical columns
print("Encoding categorical features...")

le = LabelEncoder()

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = le.fit_transform(df[column])

# Target variable
y = df["Churn Value"]

# Features
X = df.drop("Churn Value", axis=1)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy)

print("Saving model...")

joblib.dump(model, "models/churn_model.pkl")

print("Model training completed and saved.")