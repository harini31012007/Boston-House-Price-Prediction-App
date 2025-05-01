import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('model.pkl')

# UI
st.title("🏠 House Price Prediction App")
st.write("Enter the details to get the predicted price.")

# Input fields
CRIM = st.number_input('Crime Rate (CRIM)', value=0.1)
RM = st.number_input('Avg Rooms per Dwelling (RM)', value=6.0)
TAX = st.number_input('Property Tax Rate (TAX)', value=300)
PTRATIO = st.number_input('Pupil-Teacher Ratio (PTRATIO)', value=15.0)
LSTAT = st.number_input('Lower Status Population % (LSTAT)', value=5.0)

# Prediction
if st.button("Predict Price"):
    input_data = np.array([[CRIM, RM, TAX, PTRATIO, LSTAT]])
    prediction = model.predict(input_data)
    st.success(f"🏡 Predicted House Price: ${prediction[0]*1000:.2f}")
