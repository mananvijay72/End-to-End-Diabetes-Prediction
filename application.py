import streamlit as st
import pickle
import pandas as pd

# Streamlit application title
st.title("Health Parameters Input Form")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=120, value=25)
hypertension = st.selectbox("Hypertension (0: No, 1: Yes)", [0, 1])
heart_disease = st.selectbox("Heart Disease (0: No, 1: Yes)", [0, 1])
smoking_history = st.selectbox("Smoking History", ["non_smoker", "current_smoker", "past_smoker"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0, format="%.1f")
hba1c_level = st.number_input("HbA1c Level", min_value=2.0, max_value=15.0, value=5.5, format="%.1f")
blood_glucose_level = st.number_input("Blood Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, format="%.1f")

input = pd.DataFrame({"gender": [gender], "age": [age], "hypertension": [hypertension], "heart_disease": [heart_disease], "smoking_history": [smoking_history], "bmi": [bmi], "HbA1c_level": [hba1c_level], "blood_glucose_level": [blood_glucose_level]})
# Displaying the input values


# Placeholder for future model predictions or further processing
st.write("### Next Steps")
st.write("Use these inputs for further processing or prediction in your model.")

#loading model and preprocessor

with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("artifacts/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)


#function for prediction

def predict(input, model, preprocessor):
    input_data = preprocessor.transform(input)
    prediction = model.predict(input_data)
    return prediction

#prediction
prediction = predict(input, model, preprocessor)

#displaying prediction
st.write("### Prediction")
st.write(f"**Diabetes Probability:** {prediction[0]}")