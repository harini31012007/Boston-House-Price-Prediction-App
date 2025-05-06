import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Optional: LightGBM fallback-safe
try:
    from lightgbm import LGBMRegressor
    lightgbm_available = True
except ImportError:
    lightgbm_available = False

# Page Config
st.set_page_config(page_title="Boston House Price Predictor", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
    return pd.read_csv(url)

df = load_data()

# Preprocessing
scaler = StandardScaler()
X = df.drop('medv', axis=1)
y = df['medv']
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1)
}

if lightgbm_available:
    models["LightGBM"] = LGBMRegressor(n_estimators=100, learning_rate=0.1)

# UI — Prediction Interface
st.title("🏠 Boston House Price Prediction")

st.markdown("""
Customize the values of the input features below, select your preferred regression model,  
and click **Predict** to estimate the house price (in $1000s).
""")

# Input Form
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        crim = st.number_input("CRIM: Per capita crime rate", value=0.1)
        zn = st.number_input("ZN: Residential land zoned", value=0.0)
        indus = st.number_input("INDUS: Non-retail business acres", value=7.0)
        chas = st.selectbox("CHAS: Bounds Charles River", options=[0, 1])
        nox = st.number_input("NOX: Nitric oxide (ppm)", value=0.5)
        rm = st.number_input("RM: Avg rooms per dwelling", value=6.0)
        age = st.number_input("AGE: % built before 1940", value=65.0)
    with col2:
        dis = st.number_input("DIS: Distance to jobs", value=4.0)
        rad = st.number_input("RAD: Access to radial highways", value=1.0)
        tax = st.number_input("TAX: Property tax rate", value=300.0)
        ptratio = st.number_input("PTRATIO: Pupil-teacher ratio", value=18.0)
        b = st.number_input("B: 1000(Bk - 0.63)^2", value=396.9)
        lstat = st.number_input("LSTAT: % lower status population", value=12.0)

    model_choice = st.selectbox("📊 Choose Regression Model", list(models.keys()))
    submit = st.form_submit_button("🚀 Predict Price")

# Prediction
if submit:
    user_input = np.array([crim, zn, indus, chas, nox, rm, age,
                           dis, rad, tax, ptratio, b, lstat]).reshape(1, -1)
    scaled_input = scaler.transform(user_input)

    model = models[model_choice]
    model.fit(X_train, y_train)
    pred = model.predict(scaled_input)

    st.success(f"💰 Predicted House Price: **${pred[0]:.2f}k USD**")
    st.caption("Estimated using scaled inputs and selected model.")
