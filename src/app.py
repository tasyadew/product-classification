import streamlit as st
import requests
import pandas as pd

# Set page title
st.title("AEON Product Classification")

# For text prediction
user_input = st.text_input("Enter a product name:")

if st.button("Predict Text"):
    api_url = "http://api:8500/predict/" 

    payload = {"title": user_input}
    response = requests.post(api_url, json=payload)

    # Check if the request is successful
    if response.status_code == 200:
        result = response.text
        st.success(f"Predicted category: {result}")
    else:
        st.error("An error occurred while processing the request.")

# Divider
st.markdown("----")

# Upload CSV file
st.header("Prediction from CSV File")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# For CSV prediction
if st.button("Predict CSV") and uploaded_file is not None:

    api_url = "http://api:8500/predict/" 
    df = pd.read_csv(uploaded_file)
    
    # Check if the CSV file has the 'title' column
    if 'title' not in df.columns:
        st.error("CSV file must contain a 'title' column.")
    else:
        predicted_classes = []
        for title in df['title']:
            payload = {"title": title}
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                predicted_class = response.text
                predicted_classes.append(predicted_class)
            else:
                predicted_classes.append("Error")
        
        df['predicted_class'] = predicted_classes
        st.write(df)
