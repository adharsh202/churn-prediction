import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from preprocess import preprocess


# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocess data
df = preprocess(df)

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize model
model = XGBClassifier()

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/model.pkl")

# Save column names (VERY IMPORTANT)
joblib.dump(X.columns.tolist(), "model/columns.pkl")

print("✅ Model training completed and saved!")