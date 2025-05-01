import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load the saved model
model = joblib.load('model.pkl')

# Streamlit app title
st.title('House Price Prediction')

# Introduction text
st.write("""
    This web app predicts house prices based on various features from the Boston Housing Dataset.
    Simply input the required values and get a predicted house price!
""")

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    rm = st.sidebar.slider('Average number of rooms per dwelling', 3.0, 9.0, 6.0)
    lstat = st.sidebar.slider('Percentage of lower status population', 1.0, 40.0, 12.0)
    ptratio = st.sidebar.slider('Pupil-teacher ratio by town', 12.0, 22.0, 18.0)
    tax = st.sidebar.slider('Property tax rate per $10,000', 100.0, 700.0, 300.0)
    age = st.sidebar.slider('Proportion of owner-occupied units built before 1940', 1.0, 100.0, 70.0)
    
    # Creating a dictionary of user inputs
    user_data = {'rm': rm,
                 'lstat': lstat,
                 'ptratio': ptratio,
                 'tax': tax,
                 'age': age}
    
    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
input_data = user_input_features()

# Display the user inputs
st.write("User Input parameters:", input_data)

# Preprocess the data (standardizing the features)
scaler = StandardScaler()
scaled_input_data = scaler.fit_transform(input_data)

# Predict the house price using the trained model
prediction = model.predict(scaled_input_data)

# Display prediction
st.write(f"The predicted house price for the given features is: ${prediction[0]:,.2f}")

# Visualizations (Optional)
# Feature importance for Random Forest
feature_importances = model.feature_importances_

# Plotting feature importance
st.subheader('Feature Importance - Random Forest')
sns.barplot(x=feature_importances, y=input_data.columns)
plt.title('Feature Importance')
st.pyplot()

# Display correlation matrix (optional)
st.subheader('Correlation Matrix')
correlation_matrix = input_data.corr()
fig = plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
st.pyplot()
