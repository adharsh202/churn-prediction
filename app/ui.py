import streamlit as st
import requests

# API URL (local for now)
API_URL = "https://churn-prediction-x5dj.onrender.com/predict"

st.title("Customer Churn Prediction")

st.write("Enter customer details:")

# Inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", value=50.0)

# Button
if st.button("Predict"):
    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges
    }

    try:
        response = requests.post(API_URL, json=data)
        result = response.json()

        st.subheader("Prediction Result")

        if result["prediction"] == 1:
            st.error(f"⚠️ Customer likely to churn (Prob: {result['probability']:.2f})")
        else:
            st.success(f"✅ Customer likely to stay (Prob: {result['probability']:.2f})")

    except Exception as e:
        st.error("Error connecting to API")
        st.write(e)