import streamlit as st
import pandas as pd
import joblib

# Function to load model and preprocessors
def load_model_and_preprocessors(model_path, encoders_path, scaler_path, status_encoder_path):
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)
    scaler = joblib.load(scaler_path)
    status_encoder = joblib.load(status_encoder_path)
    return model, label_encoders, scaler, status_encoder

# Function to preprocess user input data
def preprocess_user_input(user_input, label_encoders, scaler):
    user_data = pd.DataFrame(user_input, index=[0])
    
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    for col in categorical_columns:
        if col in label_encoders:
            user_data[col] = label_encoders[col].transform(user_data[col])
    
    numeric_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    user_data[numeric_columns] = scaler.transform(user_data[numeric_columns])
    
    print("Processed user input data:")
    print(user_data)
    
    return user_data

# Function to predict using the model
def predict(model, user_data):
    predictions = model.predict(user_data)
    print(f"Prediction result: {predictions[0]}")
    return predictions[0]

# Load model and preprocessors
model_path = './model/random_forest_model.pkl'
encoders_path = './model/label_encoders.pkl'
scaler_path = './model/scaler.pkl'
status_encoder_path = './model/status_encoder.pkl'

model, label_encoders, scaler, status_encoder = load_model_and_preprocessors(model_path, encoders_path, scaler_path, status_encoder_path)

# Streamlit UI
st.title('Loan Eligibility Predictor')

# Input fields for user data
gender = st.selectbox('Gender', ['Male', 'Female'])
married = st.selectbox('Married', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
applicant_income = st.number_input('Applicant Income', value=0)
coapplicant_income = st.number_input('Coapplicant Income', value=0)
loan_amount = st.number_input('Loan Amount', value=0)
loan_amount_term = st.number_input('Loan Amount Term', value=0)
credit_history = st.selectbox('Credit History', [0, 1])
property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

user_input = {
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_amount_term,
    'Credit_History': credit_history,
    'Property_Area': property_area
}

# When user submits the form
if st.button('Predict'):
    # Preprocess user input
    user_data_processed = preprocess_user_input(user_input, label_encoders, scaler)
    
    # Make prediction
    prediction = predict(model, user_data_processed)
    
    # Convert prediction back to original labels
    prediction_label = status_encoder.inverse_transform([prediction])[0]
    
    # Display prediction result
    if prediction_label == 'Y':
        st.success('Congratulations! Your loan application is likely to be approved.')
    else:
        st.error('Sorry, your loan application is likely to be rejected.')
