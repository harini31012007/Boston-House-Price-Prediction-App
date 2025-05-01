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
st.set_page_config(page_title="Boston House Price Predictor", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Introduction", "Data Exploration", "Model Training", "Predict Price"])

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

# Introduction
if section == "Introduction":
    st.title("Boston House Price Prediction App")
    st.markdown("""
    Welcome! This app predicts Boston house prices using smart regression techniques.
    
    ### Models included:
    - Linear Regression
    - Ridge & Lasso Regression
    - Random Forest
    - XGBoost

    Use the sidebar to explore the dataset, train models, and make predictions!
    """)
    st.image(
        "data:image/jpeg;base64,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",
        use_column_width=True
    )

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

        chosen_model = models[model_choice]
        chosen_model.fit(X_train, y_train)
        pred = chosen_model.predict(scaled_input)

        st.success(f"🏷️ Estimated House Price: **${pred[0]:.2f}k USD**")
        st.caption("Prediction is based on selected regression model and scaled input features.")
