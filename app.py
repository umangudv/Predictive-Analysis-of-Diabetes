import streamlit as st
import pickle
import pandas as pd
import xgboost as xgb

# Load the trained model
with open('xgboost_diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the user input function
def user_input_features():
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    glucose = st.number_input('Glucose', min_value=0, max_value=200, value=85)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=122, value=66)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=29)
    insulin = st.number_input('Insulin', min_value=0, max_value=846, value=0)
    bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=26.6)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.351)
    age = st.number_input('Age', min_value=0, max_value=120, value=31)
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Display the app title
st.title('Diabetes Prediction')

# Get user input
input_df = user_input_features()

# Predict the outcome
prediction = model.predict(input_df)

# Display the result
if prediction == 1:
    st.subheader('The model predicts that the person has diabetes.')
else:
    st.subheader('The model predicts that the person does not have diabetes.')
