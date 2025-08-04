import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("smart_irrigation_model.pkl")
scaler = joblib.load("scaler.pkl")

# App title and intro
st.set_page_config(page_title="Smart Irrigation System", page_icon="💧")
st.title("💧 Smart Irrigation Prediction")
st.markdown("Use this tool to check if irrigation is needed based on environmental inputs.")

# Input sliders
temperature = st.slider("🌡️ Temperature (°C)", min_value=10.0, max_value=50.0, value=30.0)
humidity = st.slider("💧 Humidity (%)", min_value=10.0, max_value=100.0, value=60.0)
soil_moisture = st.slider("🌱 Soil Moisture (%)", min_value=0.0, max_value=100.0, value=40.0)
rainfall = st.slider("☔ Rainfall (mm)", min_value=0.0, max_value=50.0, value=5.0)

# Predict button
if st.button("🚀 Predict Irrigation Need"):
    input_data = np.array([[temperature, humidity, soil_moisture, rainfall]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ **Irrigation is needed!**")
    else:
        st.info("💤 **No irrigation needed right now.**")

