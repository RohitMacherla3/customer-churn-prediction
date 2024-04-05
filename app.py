import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title('Customer Churn Prediction')


Dependents = st.selectbox('DO you have Dependents?', ['Yes', 'No'])
Tenure_Months = int(st.number_input('Tenure'))
Online_Security = st.selectbox('Online Security', ['Yes', 'No'])
Online_Backup = st.selectbox('Online Backup', ['Yes', 'No'])
Device_Protection = st.selectbox('Device Protection', ['Yes', 'No'])
Tech_Support = st.selectbox('Tech Support', ['Yes', 'No'])
Contract_Two_year = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
Internet_Service_No = st.selectbox('Do you have Internet Service?', ['Yes', 'No'])
Internet_Service_Fiber_optic = st.selectbox('Is your Internet Service fiber optics', ['Yes', 'No'])
Monthly_Charges = st.number_input('Monthly Charges')
Total_Charges = st.number_input('Total Charges')
Payment_Method_Electronic_check = st.selectbox('Is your payment method through electronic check?', ['Yes', 'No'])

if st.button('Predict Churn'):
    model = pickle.load(open('churn_xgb_optimal.pkl', 'rb'))
    data = [[Dependents, Tenure_Months, Online_Security, Online_Backup, Device_Protection, Tech_Support, 
             Monthly_Charges, Total_Charges, Internet_Service_Fiber_optic, Internet_Service_No, Contract_Two_year, 
             Payment_Method_Electronic_check]]
    df = pd.DataFrame(data, columns=['Dependents', 'Tenure_Months', 'Online_Security', 'Online_Backup',
                                     'Device_Protection', 'Tech_Support', 'Monthly_Charges', 'Total_Charges',
                                     'Internet_Service_Fiber_optic', 'Internet_Service_No', 'Contract_Two_year', 
                                     'Payment_Method_Electronic_check'])

    categorical_feature = ['Dependents', 'Online_Security', 'Online_Backup', 'Device_Protection', 'Tech_Support',
                           'Internet_Service_Fiber_optic', 'Payment_Method_Electronic_check']
    

    for feature in categorical_feature:
        df[feature] = df[feature].apply(lambda x: 0 if x == "No" else 1)
    
    df['Contract_Two_year'] = df['Contract_Two_year'].apply(lambda x: 1 if x == 'Two year' else 0)
    df['Internet_Service_No'] = df['Internet_Service_No'].apply(lambda x: 1 if x == 'No' else 0)

    single = model.predict(df)
    probability = model.predict_proba(df)[0][single]
    probability = np.round(probability[0]*100, 2)

    if single == 1:
        op1 = "This Customer is likely to be Churned!"
        op2 = f"Confidence level is {probability}"
    else:
        op1 = "This Customer is likely to be Continue!"
        op2 = f"Confidence level is {probability}"

    st.write(op1)
    st.write(op2)