# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pickle

# Title and Description
st.set_page_config(page_title="🏠 Boston House Price Predictor", layout="centered")
st.title("🏠 Boston House Price Forecasting App")
st.markdown("Fill in the feature values below to **predict the median value of owner-occupied homes (in $1000s)** using advanced regression techniques.")

# Sidebar
st.sidebar.header("🔧 Input Features")

# Feature Inputs with Default Test Values (labels and placeholders)
crim = st.sidebar.number_input("1. CRIM - Per capita crime rate by town", 0.0, 100.0, 5.31)
zn = st.sidebar.number_input("2. ZN - % of residential land zoned for large lots", 0.0, 100.0, 11.36)
indus = st.sidebar.number_input("3. INDUS - % of non-retail business acres per town", 0.0, 30.0, 11.14)
chas = st.sidebar.selectbox("4. CHAS - Tract bounds river?", [0, 1], index=0)
nox = st.sidebar.number_input("5. NOX - Nitric oxides concentration (ppm)", 0.0, 1.0, 0.55)
rm = st.sidebar.number_input("6. RM - Average number of rooms per dwelling", 3.0, 9.0, 6.28)
age = st.sidebar.number_input("7. AGE - % of owner-occupied units built before 1940", 0.0, 100.0, 68.57)
dis = st.sidebar.number_input("8. DIS - Distances to employment centers", 1.0, 12.0, 3.80)
rad = st.sidebar.number_input("9. RAD - Accessibility to radial highways", 1.0, 24.0, 9.55)
tax = st.sidebar.number_input("10. TAX - Property-tax rate per $10,000", 100.0, 800.0, 408.24)
ptratio = st.sidebar.number_input("11. PTRATIO - Pupil-teacher ratio", 10.0, 30.0, 30.00)
b = st.sidebar.number_input("12. B - 1000(Bk - 0.63)^2", 0.0, 400.0, 353.87)
lstat = st.sidebar.number_input("13. LSTAT - % lower status of the population", 1.0, 40.0, 12.0)

# Preprocessing and Prediction
input_data = pd.DataFrame({
    'crim': [crim],
    'zn': [zn],
    'indus': [indus],
    'chas': [chas],
    'nox': [nox],
    'rm': [rm],
    'age': [age],
    'dis': [dis],
    'rad': [rad],
    'tax': [tax],
    'ptratio': [ptratio],
    'b': [b],
    'lstat': [lstat]
})

# Feature Engineering
input_data['rm_per_age'] = input_data['rm'] / input_data['age']
input_data['dis_per_rad'] = input_data['dis'] / input_data['rad']
input_data['tax_ptratio_ratio'] = input_data['tax'] / input_data['ptratio']

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(input_data[['lstat', 'rm']])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['lstat', 'rm']))
input_data['lstat^2'] = poly_df['lstat^2']
input_data['rm^2'] = poly_df['rm^2']
input_data['lstat rm'] = poly_df['lstat rm']

# Age binning
input_data['age_binned_Mid-aged'] = int(35 < age <= 70)
input_data['age_binned_Old'] = int(age > 70)

# Final feature set (same as training)
final_features = [
    'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis',
    'rad', 'tax', 'ptratio', 'b', 'lstat',
    'rm_per_age', 'dis_per_rad', 'tax_ptratio_ratio',
    'lstat^2', 'rm^2', 'lstat rm',
    'age_binned_Mid-aged', 'age_binned_Old'
]

X_input = input_data[final_features]

# Load scaler and model
@st.cache_resource
def load_model():
    # Load the scaler and model (must be saved beforehand)
    scaler = pickle.load(open("scaler.pkl", "rb"))
    model = pickle.load(open("final_model.pkl", "rb"))
    return scaler, model

scaler, model = load_model()
X_scaled = scaler.transform(X_input)

# Predict
if st.button("📊 Predict House Price"):
    prediction = model.predict(X_scaled)[0]
    st.success(f"💰 Estimated Median House Price: **${prediction * 1000:,.2f}**")

    st.markdown("---")
    st.markdown("✅ This prediction is based on advanced regression using the **Random Forest/XGBoost model** trained on the Boston Housing dataset with engineered features for improved accuracy.")

# Footer
st.markdown("---")
st.markdown("🧠 Created with love by Harini & Team • Streamlit Deployment • 2025")

