from fastapi import FastAPI
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load trained model and columns
model = joblib.load("model/model.pkl")
columns = joblib.load("model/columns.pkl")


# Home route (for testing)
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


# Prediction route
@app.post("/predict")
def predict(data: dict):
    # Convert input JSON to DataFrame
    df = pd.DataFrame([data])

    # Convert categorical to numeric
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=columns, fill_value=0)

    # Make prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }