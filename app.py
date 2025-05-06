import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Optional LightGBM import
try:
    from lightgbm import LGBMRegressor
    lightgbm_available = True
except ImportError:
    lightgbm_available = False

# Streamlit page settings
st.set_page_config(page_title="Boston House Price Predictor", layout="wide")

# Load and cache dataset
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

# Regression Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1)
}
if lightgbm_available:
    models["LightGBM"] = LGBMRegressor(n_estimators=100, learning_rate=0.1)

# Title
st.title("🏠 Boston House Price Predictor")

st.markdown("""
Fill in the house details below, select a prediction model,  
and click **Predict Price** to estimate the house's value (in $1000s).
""")

# User Input
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        crim = st.number_input("Crime Rate (per capita)", value=0.1)
        zn = st.number_input("Zoned Land (%) for Large Homes", value=0.0)
        indus = st.number_input("Industrial Area (acres)", value=7.0)
        chas = st.selectbox("Near Charles River?", options=[0, 1], help="0 = No, 1 = Yes")
        nox = st.number_input("Air Pollution (NOx level)", value=0.5)
        rm = st.number_input("Avg. Rooms per House", value=6.0)
        age = st.number_input("Old Houses (%)", value=65.0)
    with col2:
        dis = st.number_input("Distance to Employment Centers", value=4.0)
        rad = st.number_input("Highway Access Index", value=1.0)
        tax = st.number_input("Property Tax Rate", value=300.0)
        ptratio = st.number_input("Student-Teacher Ratio", value=18.0)
        b = st.number_input("Ethnicity Indicator (B)", value=396.9)
        lstat = st.number_input("Low-Income Population (%)", value=12.0)

    model_choice = st.selectbox("🔍 Choose a Prediction Model", list(models.keys()))
    submit = st.form_submit_button("🚀 Predict Price")

# Prediction logic
if submit:
    user_input = np.array([crim, zn, indus, chas, nox, rm, age,
                           dis, rad, tax, ptratio, b, lstat]).reshape(1, -1)
    scaled_input = scaler.transform(user_input)

    model = models[model_choice]
    model.fit(X_train, y_train)
    prediction = model.predict(scaled_input)

    st.success(f"🏷️ Estimated House Price: **${prediction[0]:.2f}k USD**")
    st.caption("This price is predicted using the selected machine learning model.")
