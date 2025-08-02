import streamlit as st
import joblib
import numpy as np

# Load model and features
model = joblib.load("best_aqi_model.pkl")
features = joblib.load("feature_columns.pkl")

st.title("AQI Prediction App")
st.write("Enter the input features to predict Air Quality Index (AQI)")

user_input = []
for feat in features:
    val = st.number_input(f"{feat}", format="%.4f")
    user_input.append(val)

if st.button("Predict AQI"):
    input_array = np.array(user_input).reshape(1, -1)
    pred = model.predict(input_array)[0]
    st.success(f"Predicted AQI: {pred:.2f}")
