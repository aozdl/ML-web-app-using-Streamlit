import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


# Title and description
st.title("Exercise Type Prediction")
st.write("This app predicts exercise type based on body angles.")

rf_model = joblib.load('/workspaces/ML-web-app-using-Streamlit/src/rf_model.pkl')
scaler = joblib.load('/workspaces/ML-web-app-using-Streamlit/src/scaler.pkl')

st.header("Input Body Angles")
angle1 = st.number_input("Enter angle 1")
angle2 = st.number_input("Enter angle 2")
angle3 = st.number_input("Enter angle 3")
angle4 = st.number_input("Enter angle 4")
angle5 = st.number_input("Enter angle 5")
angle6 = st.number_input("Enter angle 6")
angle7 = st.number_input("Enter angle 7")
angle8 = st.number_input("Enter angle 8")
angle9 = st.number_input("Enter angle 9")
angle10 = st.number_input("Enter angle 10")
angle11 = st.number_input("Enter angle 11")

# Create input data for prediction (with all 11 features)
input_data = np.array([[angle1, angle2, angle3, angle4, angle5, angle6, angle7, angle8, angle9, angle10, angle11]])
input_scaled = scaler.transform(input_data)


# Prediction
if st.button("Predict"):
    prediction = rf_model.predict(input_scaled)
    st.write(f"The predicted exercise type is: {prediction[0]}")
