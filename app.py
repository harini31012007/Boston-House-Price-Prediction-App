import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page config
st.set_page_config(page_title="House Price Predictor", layout="wide")

# Sidebar
st.sidebar.title("🔎 Navigation")
section = st.sidebar.radio("Go to", ["Introduction", "Data Exploration", "Model Training", "Predict Price"])

# Load dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
    return pd.read_csv(url)

df = load_data()

# Standard scaler
scaler = StandardScaler()
X = df.drop('medv', axis=1)
y = df['medv']
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1)
}

# Introduction
if section == "Introduction":
    st.title("Boston House Price Prediction App")
    st.markdown("""
    Welcome! This app helps predict Boston house prices using smart regression techniques.
    
    Models used:
    - Linear Regression
    - Ridge & Lasso Regression
    - Random Forest
    - XGBoost
    
     You can explore data, train models, and predict your own house prices using this app.
    """)

    st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*RBPVDn3S1sauz1MyoN7Hug.jpeg", use_column_width=True)

# Data Exploration
elif section == "Data Exploration":
    st.title("Exploratory Data Analysis")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution of Target Variable (medv)")
    fig2, ax2 = plt.subplots()
    sns.histplot(df["medv"], kde=True, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Missing Values & Duplicates")
    st.write("Missing Values:")
    st.write(df.isnull().sum())
    st.write("Duplicates:", df.duplicated().sum())

# Model Training
elif section == "Model Training":
    st.title("Model Training & Evaluation")

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "R²": r2_score(y_test, preds),
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds))
        }

    st.subheader("Performance Comparison")

    col1, col2, col3 = st.columns(3)
    best_model = max(results, key=lambda x: results[x]["R²"])
    
    col1.metric("Best Model", best_model)
    col2.metric("Highest R²", f"{results[best_model]['R²']:.3f}")
    col3.metric("Lowest RMSE", f"{results[best_model]['RMSE']:.2f}")

    st.markdown("### Detailed Metrics")
    result_df = pd.DataFrame(results).T.sort_values(by="R²", ascending=False)
    st.dataframe(result_df.style.background_gradient(cmap='Greens'))

# Prediction
elif section == "Predict Price":
    st.title("Predict House Price")

    st.markdown("Fill the details below to predict house price.")

    columns = df.columns[:-1]
    user_input = []
    for col in columns:
        val = st.number_input(f"{col}", value=float(df[col].mean()), step=0.1)
        user_input.append(val)

    input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(input_array)

    model_choice = st.selectbox("Choose a Model", list(models.keys()))
    chosen_model = models[model_choice]
    chosen_model.fit(X_train, y_train)
    pred = chosen_model.predict(scaled_input)

    st.success(f"Predicted House Price (medv): ${pred[0]:.2f}")

    st.markdown("---")
    st.info("Note: medv is in $1000s.")

