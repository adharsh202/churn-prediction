import pandas as pd

def preprocess(df):
    # Create a copy to avoid modifying original data
    df = df.copy()

    # Remove unnecessary column
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Convert target column to numeric
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Convert categorical variables into numeric
    df = pd.get_dummies(df, drop_first=True)

    return df