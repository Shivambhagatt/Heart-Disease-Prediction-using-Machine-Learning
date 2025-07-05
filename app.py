import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Header with image
st.image("https://i.imgur.com/2XJZz6a.png", width=100)
st.title("🫀 Heart Disease Prediction App")
st.markdown("Predict the risk of heart disease using patient information.")

st.markdown("---")
st.subheader("📋 Enter Patient Details")

# Input form layout
with st.form("heart_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 30)
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
        chol = st.number_input("Cholesterol (chol)", 100, 400, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])

    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        thalach = st.number_input("Max Heart Rate (thalach)", 70, 210, 150)
        exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (thal)", [1, 2, 3])

    submit = st.form_submit_button("🔍 Predict")

if submit:
    # Data processing
    sex = 1 if sex == "Male" else 0
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    
    prediction = model.predict(features)

    st.markdown("---")
    if prediction[0] == 1:
        st.error("⚠️ The patient is **at risk** of heart disease.")
    else:
        st.success("✅ The patient is **not likely** to have heart disease.")

# Sidebar
st.sidebar.title("🧠 About")
st.sidebar.info("Built by Shivam using Streamlit + Machine Learning")
st.sidebar.markdown("[GitHub Repo](https://github.com/Shivambhagatt/Heart-Disease-Prediction-using-Machine-Learning)")
