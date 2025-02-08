import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px 
import warnings
warnings.filterwarnings('ignore') 

# Load the dataset
data = pd.read_csv('Loan_Data.csv')

# Title
st.markdown("<h1 style='color: #780C28; font-size: 36px ; font-family: helvetica; text-align: center;'>Loan Data Analytics</h1>", unsafe_allow_html = True)

st.markdown("<h3 style='margin-top: -20px; color: #1D1616; font-size: 24px ; font-family: helvetica; text-align: center;'>Built by Lazy IDee</h3>", unsafe_allow_html = True)

# Images to be displayed
st.image('HomePic.png')
st.sidebar.image('SideBar.png')

# Input fields for user to provide details
st.sidebar.header('User Input Parameters')

# st.markdown("<br><br>", unsafe_allow_html = True)
# st.dataframe(data, use_container_width = True)

st.divider()

# Display the dataset
applicant_income = st.sidebar.number_input('Applicant Income', min_value=0)
coapplicant_income = st.sidebar.number_input('Coapplicant Income', min_value=0)
loan_amount = st.sidebar.number_input('Loan Amount', min_value=0)
dependents = st.sidebar.selectbox('Dependents', data.Dependents.unique()) 
property_area = st.sidebar.selectbox('Property Area', data['Property_Area'].unique()) 
credit_history = st.sidebar.selectbox('Credit History', data['Credit_History'].unique())
    
users_input = pd.DataFrame({'ApplicantIncome': [applicant_income], 'LoanAmount': [loan_amount], 'CoapplicantIncome': [coapplicant_income], 'Dependents': [dependents], 'Property_Area': [property_area], 'Credit_History': [credit_history]})

# Display user input
st.header("User Input Parameters")
st.dataframe(users_input)

import joblib 

# Step 1: Load the pre-trained model and scalers
app_transformer = joblib.load('ApplicantIncome_scaler.pkl')
coapp_transformer = joblib.load('CoapplicantIncome_scaler.pkl')
loan_transformer = joblib.load('LoanAmount_scaler.pkl')
dep_transformer = joblib.load('Dependents_scaler.pkl')  
prop_transformer = joblib.load('Property_Area_encoder.pkl')  
cred_transformer = joblib.load('Credit_History_scaler.pkl')  

# Step 2: Transform the input variables
users_input['ApplicantIncome'] = app_transformer.transform(users_input[['ApplicantIncome']])
users_input['LoanAmount'] = loan_transformer.transform(users_input[['LoanAmount']])
users_input['CoapplicantIncome'] = coapp_transformer.transform(users_input[['CoapplicantIncome']])
users_input['Dependents'] = dep_transformer.transform(users_input[['Dependents']])
users_input['Property_Area'] = prop_transformer.transform(users_input[['Property_Area']])
users_input['Credit_History'] = cred_transformer.transform(users_input[['Credit_History']])

# Step 3: Load the model and make predictions
model = joblib.load('loan_model.pkl')  

predictButton = st.button("Push to Predict")

if predictButton:
    prediction = model.predict(users_input)
    if prediction[0] == 1:
        st.success("Congratulations! Your loan is likely to be approved.")
    else:
        st.error("Unfortunately, your loan is likely to be rejected.")