import streamlit as st
import pickle
import numpy as np
from PIL import Image
import json

# Path to the trained model
#model_path = 'model.pkl'
model_path = 'deployment/web/model.pkl'

# Load the trained Logistic Regression model
with open(model_path, 'rb') as f_in: 
    dv, model = pickle.load(f_in)

# Set page configuration
st.set_page_config(page_title="Stroke Prediction App", layout="centered")

# Add a custom background style
def set_background():
    st.markdown(
        """
        <style>
        body {
            background-color: #326ba8;  /* Light medical blue */
            color: #003366;  /* Dark blue text */
        }
        .stApp {
            background-color: #326ba8;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# Add a header with an image
st.title("🩺 Stroke Prediction App")
#image = Image.open("rod.jpg")  # Replace with your image file
#st.image(image, use_column_width=True)

st.write(
    """
    This app uses a Logistic Regression model to predict the likelihood of a stroke based on patient data. 
    Enter the required details below to get the prediction.
    """
)

# Input fields for patient data
st.sidebar.header("Patient Information")

user_inputs = {
    "gender": st.sidebar.selectbox("Gender", options=["male", "female", "other"]),
    "age": st.sidebar.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1),
    "hypertension": st.sidebar.selectbox("Hypertension", options=["no", "yes"]),
    "heart_disease": st.sidebar.selectbox("Heart Disease", options=["no", "yes"]),
    "ever_married": st.sidebar.selectbox("Ever Married", options=["no", "yes"]),
    "work_type": st.sidebar.selectbox("Work Type", options=["children", "govt_job", "never_worked", "private", "self-employed"]),
    "residence_type": st.sidebar.selectbox("Resident Type", options=["rural", "urban"]),
    "avg_glucose_level": st.sidebar.number_input("Average Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=100.0),
    "bmi": st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0),
    "smoking_status": st.sidebar.selectbox(
        "Smoking Status",
        options=["never_smoked", "formerly_smoked", "smokes", "unknown"]
    ),
}

# Convert dictionary to JSON string
#user_inputs_json = json.dumps(user_inputs, indent=4)


# Transform the JSON data using DictVectorizer
X = dv.transform([user_inputs])

# Predict button
if st.sidebar.button("Predict"):
    # Make prediction using the loaded model
    prediction = model.predict(X)[0]
    prediction_proba = model.predict_proba(X)[0][1]  # Probability of stroke (positive class)
    
    # Display results
    st.subheader("Prediction Results")
    if prediction == 1:
        st.error("🚨 The patient is at risk of having a stroke.")
    else:
        st.success("✅ The patient is not at risk of having a stroke.")
    
    st.write(f"**Prediction Probability:** {prediction_proba:.2%}")

# Footer
st.markdown("---")
st.markdown(
    """
    *Note: This prediction is based on statistical modeling and should not replace clinical diagnosis. 
    Consult a healthcare professional for an accurate evaluation.*
    """
)

