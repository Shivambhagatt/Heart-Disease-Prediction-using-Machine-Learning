import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('heart_disease_model.pkl', 'rb'))

# Page title
st.markdown("<h1 style='text-align: center; color: #333;'>💓 Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("#### Enter the patient details below:")

# Optional CSS
st.markdown("""
    <style>
        .stNumberInput input, .stSelectbox div {
            background-color: #f9f9f9;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# User Inputs in 2 columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 30)
    sex = st.selectbox("Sex", ['Male', 'Female'])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])

with col2:
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol = st.number_input("Cholesterol (chol)", 100, 400, 200)
    thalach = st.number_input("Max Heart Rate (thalach)", 60, 210, 150)
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Preprocess input
sex = 1 if sex == 'Male' else 0

features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

# Predict
if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("🚨 High risk of Heart Disease!")
    else:
        st.success("✅ Low risk of Heart Disease.")
