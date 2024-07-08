import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("landslide_detection_model.h5")

# Function to preprocess input data
# Function to preprocess input data
def preprocess_input(input_data):
    # Preprocess the input data using the same scaler used during training
    scaler = StandardScaler()
    # Assuming input_data is a list or array with 5 elements
    scaled_input = scaler.fit_transform(np.array([input_data]))  # Reshape to (1, 5)
    return scaled_input


# Streamlit app
def main():
    # Set page title
    st.title("Landslide Detection")

    # Create input fields for rainfall parameters
    st.header("Enter Rainfall Parameters")
    rainfall_duration = st.number_input("Rainfall Duration", min_value=0.0, step=0.1)
    rainfall_intensity = st.number_input("Rainfall Intensity", min_value=0.0, step=0.1)
    rainfall_accumulation = st.number_input("Rainfall Accumulation", min_value=0.0, step=0.1)
    soil_moisture = st.number_input("Soil Moisture", min_value=0.0, step=0.1)

    # Preprocess input data
    input_data = np.array([[rainfall_duration, rainfall_intensity, rainfall_accumulation, soil_moisture]])
    scaled_input = preprocess_input(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)

    # Display prediction result
    st.header("Prediction")
    if prediction > 0.5:
        st.write("Based on the provided data, a landslide is likely to occur.")
    else:
        st.write("Based on the provided data, no landslide is likely to occur.")

if __name__ == "__main__":
    main()

