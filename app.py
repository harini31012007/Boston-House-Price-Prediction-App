import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

# Set Streamlit config
st.set_page_config(page_title="Boston House Price Predictor", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
    return pd.read_csv(url)

df = load_data()

# Preprocess
scaler = StandardScaler()
poly = PolynomialFeatures(degree=2, include_bias=False)

X = df.drop('medv', axis=1)
y = df['medv']
X_scaled = scaler.fit_transform(X)
X_poly = poly.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1),
    "LightGBM": LGBMRegressor(n_estimators=100),
    "CatBoost": CatBoostRegressor(verbose=0)
}

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.success("🔮 Prediction Mode")

# Main Section - Prediction
st.title("🏠 Boston House Price Prediction")

st.markdown("#### Enter the property details:")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        crim = st.number_input("CRIM: Per capita crime rate", value=5.31, step=0.1)
        zn = st.number_input("ZN: Residential zoning", value=11.36, step=0.1)
        indus = st.number_input("INDUS: Non-retail business acres", value=11.14, step=0.1)
        chas = st.number_input("CHAS: Charles River (1 = yes, 0 = no)", value=0.0, step=1.0)
        nox = st.number_input("NOX: Nitric oxide concentration", value=0.55, step=0.01)
        rm = st.number_input("RM: Average rooms per dwelling", value=6.28, step=0.1)
        age = st.number_input("AGE: Older buildings %", value=68.57, step=0.1)
    with col2:
        dis = st.number_input("DIS: Distance to jobs", value=3.80, step=0.1)
        rad = st.number_input("RAD: Access to highways", value=9.55, step=0.1)
        tax = st.number_input("TAX: Property tax rate", value=408.24, step=0.1)
        ptratio = st.number_input("PTRATIO: Student-teacher ratio", value=18.0, step=0.1)
        b = st.number_input("B: 1000(Bk - 0.63)^2", value=353.87, step=0.1)
        lstat = st.number_input("LSTAT: % lower status population", value=12.65, step=0.1)

    model_choice = st.selectbox("🧠 Choose a Model", list(models.keys()))
    submitted = st.form_submit_button("🚀 Predict Now")

if submitted:
    # Get and transform input
    user_input = np.array([crim, zn, indus, chas, nox, rm, age,
                           dis, rad, tax, ptratio, b, lstat]).reshape(1, -1)
    scaled_input = scaler.transform(user_input)
    poly_input = poly.transform(scaled_input)

    # Train & predict
    model = models[model_choice]
    model.fit(X_train, y_train)
    prediction = model.predict(poly_input)

    # Show results
    st.success(f"🏷️ Predicted House Price: **${prediction[0]:.2f}k USD**")

    # Show model performance
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    st.info(f"📊 Model R² Score on Test Set: **{r2:.3f}**")
