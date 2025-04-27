import streamlit as st
import numpy as np
import joblib

# Load the saved model, scaler, and label encoder
model = joblib.load('model (1).pkl')
scaler = joblib.load('scaler.pkl')
risk_category_encoder = joblib.load('risk_category_encoder.pkl')

st.title("🏥 Patient Risk Category Prediction")
st.write("Please enter the following patient details:")

# Input widgets
heart_rate = st.number_input('Heart Rate', min_value=30, max_value=200, value=70)
respiratory_rate = st.number_input('Respiratory Rate', min_value=10, max_value=50, value=16)
body_temperature = st.number_input('Body Temperature (°C)', min_value=30.0, max_value=45.0, value=37.0)
oxygen_saturation = st.number_input('Oxygen Saturation (%)', min_value=60, max_value=100, value=98)
systolic_bp = st.number_input('Systolic Blood Pressure (mmHg)', min_value=80, max_value=200, value=120)
diastolic_bp = st.number_input('Diastolic Blood Pressure (mmHg)', min_value=40, max_value=120, value=80)
age = st.number_input('Age (years)', min_value=0, max_value=120, value=30)
gender = st.selectbox('Gender', ['Female', 'Male'])
weight = st.number_input('Weight (kg)', min_value=10.0, max_value=200.0, value=70.0)
height = st.number_input('Height (m)', min_value=0.5, max_value=2.5, value=1.7)

# **Derived Values Calculations**:

# Pulse Pressure: Systolic BP - Diastolic BP
derived_pp = systolic_bp - diastolic_bp

# BMI: Weight (kg) / (Height (m))^2
derived_bmi = weight / (height ** 2)

# Mean Arterial Pressure (MAP): Diastolic BP + 1/3(Systolic BP - Diastolic BP)
derived_map = diastolic_bp + (1/3) * (systolic_bp - diastolic_bp)

# HRV (example, replace with actual calculation if needed)
# For simplicity, we will set it as a function of heart rate. Modify as per your formula.
derived_hrv = 100 / heart_rate  # Example calculation, replace with real formula

if st.button('Predict Risk Category'):
    # Encode gender the same way as training
    gender_encoded = 1 if gender == 'Male' else 0
    
    # Build the feature vector in the exact order used during training
    input_data = np.array([[
        heart_rate,
        respiratory_rate,
        body_temperature,
        oxygen_saturation,
        systolic_bp,
        diastolic_bp,
        age,
        gender_encoded,
        weight,
        height,
        derived_hrv,
        derived_pp,
        derived_bmi,
        derived_map
    ]])

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    category = risk_category_encoder.inverse_transform(prediction)[0]

    st.success(f"🔍 Predicted Risk Category: **{category}**")
