import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.title("Diabetes Prediction")

model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

preg = st.number_input("Pregnancies", 0, 20, 2)
glu = st.number_input("Glucose", 0, 300, 120)
bp = st.number_input("BloodPressure", 0, 200, 70)
skin = st.number_input("SkinThickness", 0, 100, 20)
ins = st.number_input("Insulin", 0, 1000, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 30.0)
dpf = st.number_input("DiabetesPedigreeFunction", 0.0, 3.0, 0.4)
age = st.number_input("Age", 0, 120, 33)

if st.button("Predict"):
    x = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    x_scaled = scaler.transform(x)

    pred = model.predict(x_scaled)[0]
    probs = model.predict_proba(x_scaled)[0]

    st.write("Result:")
    st.write("Diabetic" if pred == 1 else "Not Diabetic")

    prob_df = pd.DataFrame(
        {"class": ["Not Diabetic (0)", "Diabetic (1)"], "probability": probs}
    ).set_index("class")

    st.bar_chart(prob_df)
