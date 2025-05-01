import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set the page title and layout for the Streamlit app
st.set_page_config(page_title="House Price Prediction", layout="wide")

# App Title
st.title("House Price Prediction Using Smart Regression Techniques")

# Introduction Text
st.markdown("""
This app predicts house prices using various regression models, including Linear Regression, Ridge Regression, Lasso Regression, Random Forest, and XGBoost. The dataset used is the Boston Housing dataset.
""")

# Dataset Loading
st.subheader('Dataset Overview')
url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
df = pd.read_csv(url)
st.write("Initial Dataset Loaded.")
st.dataframe(df.head())

# Data Cleaning and Preparation
st.subheader('Data Cleaning')
st.write("\nMissing Values Check:\n", df.isnull().sum())
duplicates = df.duplicated().sum()
st.write(f"Number of duplicate records: {duplicates}")

df.drop_duplicates(inplace=True)

# Visualize Outliers using Boxplots
st.subheader('Outlier Detection - Boxplots')
fig, axes = plt.subplots(4, 4, figsize=(15, 10))
for i, col in enumerate(df.columns[:-1], 1):
    ax = axes[(i-1)//4, (i-1)%4]
    sns.boxplot(data=df, x=col, ax=ax)
plt.tight_layout()
st.pyplot(fig)

# Feature Engineering (Polynomial Features - Example)
st.subheader('Feature Engineering - Polynomial Features')
degree = st.slider('Select Degree of Polynomial Features', 1, 3, 1)
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(df.drop('medv', axis=1))
st.write("Polynomial features generated with degree", degree)

# Exploratory Data Analysis (EDA)
st.subheader('Exploratory Data Analysis (EDA)')
st.write("Check the distribution of features.")
fig, ax = plt.subplots(figsize=(16, 12))
df.hist(ax=ax, bins=30, edgecolor='black')
st.pyplot(fig)

st.subheader('Correlation Matrix')
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)

# Feature-Target Split
st.subheader('Data Splitting')
X = df.drop("medv", axis=1)
y = df["medv"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models for evaluation
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1)
}

# Train and evaluate models
st.subheader('Model Training and Evaluation')
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    results[name] = {
        "R² Score": r2_score(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds))
    }

# Display model performance comparison
st.write("\nModel Performance Comparison:")
for model_name, metrics in results.items():
    st.write(f"{model_name} --> R²: {metrics['R² Score']:.4f}, MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}")

# Visualize Results (Residual Plot - Random Forest as an example)
st.subheader('Residual Plot - Random Forest')
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

residuals = y_test - y_pred_rf
fig, ax = plt.subplots()
ax.scatter(y_pred_rf, residuals, alpha=0.6, color='blue')
ax.axhline(0, color='red', linewidth=2)
ax.set_xlabel('Predicted Prices')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot - Random Forest')
st.pyplot(fig)

# Prediction Form (User Input)
st.subheader('Make a Prediction')
st.write("""
Use the form below to input house features and predict the house price.
""")

# Input fields for user input
crim = st.number_input('Crime Rate (crim)', value=0.1, min_value=0.0)
zn = st.number_input('Residential Land Zoned (zn)', value=0.0, min_value=0.0)
indus = st.number_input('Non-Retail Business Acres (indus)', value=10.0, min_value=0.0)
chas = st.selectbox('Charles River Dummy Variable (chas)', [0, 1])
nox = st.number_input('Nitrogen Oxides Concentration (nox)', value=0.5, min_value=0.0)
rm = st.number_input('Average Number of Rooms (rm)', value=6.0, min_value=0.0)
age = st.number_input('Proportion of Older Homes (age)', value=50.0, min_value=0.0)
dis = st.number_input('Distance to Employment Centers (dis)', value=4.0, min_value=0.0)
rad = st.number_input('Index of Accessibility to Radial Highways (rad)', value=4.0, min_value=0.0)
tax = st.number_input('Property Tax Rate (tax)', value=330.0, min_value=0.0)
ptratio = st.number_input('Pupil-Teacher Ratio (ptratio)', value=18.0, min_value=0.0)
black = st.number_input('Black Population (black)', value=390.0, min_value=0.0)
lstat = st.number_input('Percentage of Lower Status Population (lstat)', value=12.0, min_value=0.0)

user_input = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat]])
user_input_scaled = scaler.transform(user_input)

# Making Prediction with Random Forest
prediction = rf.predict(user_input_scaled)
st.write(f"Predicted House Price: ${prediction[0]:,.2f}")

