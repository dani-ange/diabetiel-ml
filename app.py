import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("diabetes_model.joblib")

# Streamlit UI
st.title("Diabetes Prediction App")
st.write("Enter patient data to predict diabetes")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0)
age = st.number_input("Age", min_value=0, max_value=120)

# Prediction
if st.button("Predict"):
    features = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
    prediction = model.predict(features)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    st.write(f"Prediction: **{result}**")
