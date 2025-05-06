import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Set page config
st.set_page_config(page_title="Boston House Price Predictor", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
    return pd.read_csv(url)

df = load_data()

# Preprocess
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

# Sidebar Navigation - Only Prediction Enabled
st.sidebar.title("Navigation")
st.sidebar.info("🔮 Only Prediction Enabled")
section = "Predict Price"

# Prediction Section
if section == "Predict Price":
    st.title("Predict Boston House Price")

    st.markdown("""
    ### Enter House Features
    Provide values for each feature to get a predicted house price.  
    *(Note: All features are numeric. `medv` is the target variable and is excluded.)*
    """)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            crim = st.number_input("CRIM: Per capita crime rate by town", value=5.31, step=0.1)
            zn = st.number_input("ZN: Proportion of residential land zoned for large lots", value=11.36, step=0.1)
            indus = st.number_input("INDUS: Proportion of non-retail business acres per town", value=11.14, step=0.1)
            chas = st.number_input("CHAS: Charles River dummy variable (1 = yes; 0 = no)", value=0.07, step=0.01)
            nox = st.number_input("NOX: Nitric oxide concentration (ppm)", value=0.55, step=0.01)
            rm = st.number_input("RM: Average number of rooms per dwelling", value=6.28, step=0.1)
            age = st.number_input("AGE: % of owner-occupied units built before 1940", value=68.57, step=0.1)
        with col2:
            dis = st.number_input("DIS: Distance to employment centres", value=3.80, step=0.1)
            rad = st.number_input("RAD: Accessibility to radial highways", value=9.55, step=0.1)
            tax = st.number_input("TAX: Property tax rate per $10,000", value=408.24, step=0.1)
            ptratio = st.number_input("PTRATIO: Pupil-teacher ratio", value=30.00, step=0.1)
            b = st.number_input("B: 1000(Bk - 0.63)^2 (Bk = % Black population)", value=353.87, step=0.1)
            lstat = st.number_input("LSTAT: % lower status of population", value=12.65, step=0.1)

        model_choice = st.selectbox("🧠 Choose a Regression Model", list(models.keys()))
        submitted = st.form_submit_button("🚀 Predict Price")

    if submitted:
        user_input = [crim, zn, indus, chas, nox, rm, age,
                      dis, rad, tax, ptratio, b, lstat]
        input_array = np.array(user_input).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        # Train the chosen model
        chosen_model = models[model_choice]
        chosen_model.fit(X_train, y_train)
        pred = chosen_model.predict(scaled_input)

        st.success(f"🏷️ Estimated House Price: **${pred[0]:.2f}k USD**")
        st.caption("Prediction is based on selected regression model and scaled input features.")
