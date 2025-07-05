import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

# Preprocessing
X = df.drop("target", axis=1)
y = df["target"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Streamlit UI
st.markdown(
    "<h1 style='text-align: center; color: #333;'>❤️&nbsp;Heart Disease Prediction App</h1>",
    unsafe_allow_html=True
)
st.write("Enter the patient details below:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", value=120)
chol = st.number_input("Cholesterol (chol)", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (fbs)", options=[0, 1])
restecg = st.selectbox("Rest ECG (restecg)", options=[0, 1, 2])
thalach = st.number_input("Max Heart Rate (thalach)", value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])
oldpeak = st.number_input("Oldpeak", value=1.0)
slope = st.selectbox("Slope", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3])
thal = st.selectbox("Thal", options=[0, 1, 2, 3])

# Predict button
if st.button("Predict"):
    user_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(user_data)
    result = "🟢 No Heart Disease" if prediction[0] == 0 else "🔴 Heart Disease Detected"
    st.success(result)
