import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open('rainfall_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the application
st.title("Rainfall Prediction Model")
st.write("Forecast the total rainfall for the next 10 years based on historical data.")

# Input features for forecasting
st.subheader("Input Data for Forecast")
years = st.slider("Enter the number of future years to forecast:", 1, 10)

# Forecast button
if st.button("Forecast"):
    # Dummy feature preparation (replace with actual preprocessing if needed)
    features = np.array([[i] for i in range(1, years + 1)])  # Ensure features are correctly shaped
    predictions = model.predict(features)
    predictions = predictions.flatten()  # Ensure predictions are 1-dimensional
    
    # Create a DataFrame
    forecast_df = pd.DataFrame({
        "Year": range(1, years + 1),
        "Predicted Rainfall": predictions
    })
    
    # Display results
    st.write("Forecasted rainfall for the next {} years:".format(years))
    st.dataframe(forecast_df)
    
    # Visualization
    st.line_chart(forecast_df.set_index("Year"))