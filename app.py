import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.markdown(
    """
    <h1 style='text-align: center;'>Customer Churn Prediction</h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align: center;'>Check all that apply to the customer</p>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)
Dependents = col1.checkbox('Dependents')
Online_Security = col2.checkbox('Online Security')
Online_Backup = col3.checkbox('Online Backup')

col4, col5, col6 = st.columns(3)
Tech_Support = col4.checkbox('Tech Support')
Device_Protection = col5.checkbox('Device Protection')
Internet_Service_No = col6.checkbox('Internet Service')

col7, col8 = st.columns(2)
Payment_Method_Electronic_check = col7.checkbox('Payment method is electronic check')
Internet_Service_Fiber_optic = col8.checkbox('Internet Service is fiber optics')

col9, col10 = st.columns(2)
Monthly_Charges = col9.number_input('Monthly Charges')
Total_Charges = col10.number_input('Total Charges')

Contract_Two_year = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
Tenure_Months = st.number_input('Tenure')

st.markdown(
        """
        <style>
        /* Increase the size of the button */
        .stButton>button {
            width: 200px; 
            height: 50px; 
            margin: 0 auto;
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
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
        df[feature] = df[feature].apply(lambda x: 1 if x else 0)
    
    df['Contract_Two_year'] = df['Contract_Two_year'].apply(lambda x: 1 if x == 'Two year' else 0)
    df['Internet_Service_No'] = df['Internet_Service_No'].apply(lambda x: 0 if x else 1)

    single = model.predict(df)
    probability = model.predict_proba(df)[0][single]
    probability = np.round(probability[0]*100, 2)

    if single == 1:
        op1 = "This Customer is likely to be Churned :("
        op2 = f"Confidence level is {probability}"
    else:
        op1 = "This Customer is likely to be Continue!"
        op2 = f"Confidence level is {probability}"

    centered_content_1 = f'<div style="text-align:center">{op1}</div>'
    centered_content_2 = f'<div style="text-align:center">{op2}</div>'

    st.write(centered_content_1, unsafe_allow_html=True)
    st.write(centered_content_2, unsafe_allow_html=True)