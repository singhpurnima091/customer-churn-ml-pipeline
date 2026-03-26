import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv("data/processed.csv")

# -------------------------------
# Drop unnecessary columns
# -------------------------------
df = df.drop([
    "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude"
], axis=1)

# -------------------------------
# Target & Features
# -------------------------------
y = df["Churn Value"]

X = df.drop([
    "Churn Value", "Churn Label",
    "Churn Score", "CLTV", "Churn Reason"
], axis=1)

# -------------------------------
# Feature groups
# -------------------------------
num_features = ["Tenure Months", "Monthly Charges", "Total Charges"]
cat_features = [col for col in X.columns if col not in num_features]

# -------------------------------
# Pipelines
# -------------------------------
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# -------------------------------
# Final Pipeline
# -------------------------------
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000))
])

# -------------------------------
# Train
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# -------------------------------
# Save
# -------------------------------
joblib.dump(pipeline, "models/churn_pipeline.pkl")

print("✅ Model trained & saved successfully!")