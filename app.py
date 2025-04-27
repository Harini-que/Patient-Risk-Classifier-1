import streamlit as st
import numpy as np
import joblib

# Load the saved model, scaler, and label encoder
model = joblib.load('model (1).pkl')
scaler = joblib.load('scaler.pkl')
risk_category_encoder = joblib.load('risk_category_encoder.pkl')

st.title("🏥 Patient Risk Category Prediction")

st.write("Please enter the following patient details:")

# Create input fields for each feature in your dataset
heart_rate = st.number_input('Heart Rate', min_value=30, max_value=200, value=70)
respiratory_rate = st.number_input('Respiratory Rate', min_value=10, max_value=50, value=16)
body_temperature = st.number_input('Body Temperature (°C)', min_value=30, max_value=45, value=37)
oxygen_saturation = st.number_input('Oxygen Saturation (%)', min_value=60, max_value=100, value=98)
systolic_blood_pressure = st.number_input('Systolic Blood Pressure (mmHg)', min_value=80, max_value=200, value=120)
diastolic_blood_pressure = st.number_input('Diastolic Blood Pressure (mmHg)', min_value=40, max_value=120, value=80)
age = st.number_input('Age (years)', min_value=0, max_value=120, value=30)
gender = st.selectbox('Gender', ['Female', 'Male'])
weight = st.number_input('Weight (kg)', min_value=10, max_value=200, value=70)
height = st.number_input('Height (m)', min_value=0.5, max_value=2.5, value=1.7)
derived_hrv = st.number_input('Derived HRV', min_value=0.0, max_value=200.0, value=50.0)
derived_pulse_pressure = st.number_input('Derived Pulse Pressure', min_value=10.0, max_value=100.0, value=40.0)
derived_bmi = st.number_input('Derived BMI', min_value=10.0, max_value=50.0, value=25.0)
derived_map = st.number_input('Derived MAP', min_value=40.0, max_value=150.0, value=90.0)

# Button to predict
if st.button('Predict Risk Category'):
    # Encode Gender (same way as in training: Female=0, Male=1)
    gender_encoded = 1 if gender == 'Male' else 0
    
    # Prepare the feature array based on all the inputs
    input_data = np.array([[heart_rate, respiratory_rate, body_temperature, oxygen_saturation,
                            systolic_blood_pressure, diastolic_blood_pressure, age, gender_encoded, 
                            weight, height, derived_hrv, derived_pulse_pressure, derived_bmi, derived_map]])

    # Scale the features using the same scaler as the model
    input_scaled = scaler.transform(input_data)

    # Predict the risk category
    prediction = model.predict(input_scaled)
    predicted_category = risk_category_encoder.inverse_transform(prediction)[0]

    st.success(f"The predicted Risk Category is: **{predicted_category}** 🎯")
