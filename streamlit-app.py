import streamlit as st
import requests

# Define the endpoint
endpoint = "http://localhost:8000/predict"

# Create a text input
text = st.text_input("Enter text to classify")

# Create a submit button
if st.button("Submit"):
    # Send a POST request to the endpoint with the text
    response = requests.post(endpoint, json={"text": text})
    # Get the JSON response
    response_json = response.json()
    # Extract the label from the response
    label = response_json["label"]
    # Display the label
    st.success(f"The predicted class is: {label}")
