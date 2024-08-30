import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the trained model and scaler
model = joblib.load('models/predictive_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Streamlit app
st.title('Predictive Maintenance Model')

# Input features
st.sidebar.header('Input Features')
input_data = {}
for feature in pd.read_csv('data/sensor_data.csv').columns[:-1]:  # Exclude the target column
    if feature != 'timestamp':
        input_data[feature] = st.sidebar.number_input(feature, value=0.0)
    else:
        input_data[feature] = st.sidebar.text_input(feature, value='01-01-2024 05:00')

# Predict button
if st.sidebar.button('Predict'):
    input_df = pd.DataFrame([input_data])
    
    # Convert timestamp to datetime and extract features
    input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
    input_df['year'] = input_df['timestamp'].dt.year
    input_df['month'] = input_df['timestamp'].dt.month
    input_df['day'] = input_df['timestamp'].dt.day
    input_df['hour'] = input_df['timestamp'].dt.hour
    input_df['minute'] = input_df['timestamp'].dt.minute
    input_df = input_df.drop('timestamp', axis=1)
    
    # Standardize features
    input_df = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_df)
    st.write(f'Prediction: {"Failure" if prediction[0] else "No Failure"}')

    # Display input data for verification
    st.write("Input Data:")
    st.write(input_data)
