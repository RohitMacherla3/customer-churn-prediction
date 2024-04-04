import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

st.title('Customer Churn Prediction')


Dependents = st.selectbox('Dependents', ['Yes', 'No'])
tenure = st.number_input('tenure')
OnlineSecurity = st.selectbox('OnlineSecurity', ['Yes', 'No'])
OnlineBackup = st.selectbox('OnlineBackup', ['Yes', 'No'])
DeviceProtection = st.selectbox('DeviceProtection', ['Yes', 'No'])
TechSupport = st.selectbox('TechSupport', ['Yes', 'No'])
Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox('PaperlessBilling', ['Yes', 'No'])
MonthlyCharges = st.number_input('MonthlyCharges')
TotalCharges = st.number_input('TotalCharges')

model = pickle.load(open('churn_xgb_optimal.pkl', 'rb'))
data = [[Dependents, tenure, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, Contract, PaperlessBilling, MonthlyCharges, TotalCharges]]
df = pd.DataFrame(data, columns=['Dependents', 'tenure', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract',
    'PaperlessBilling', 'MonthlyCharges', 'TotalCharges'])

categorical_feature = {feature for feature in df.columns if df[feature].dtypes == 'O'}

encoder = LabelEncoder()
for feature in categorical_feature:
    df[feature] = encoder.fit_transform(df[feature])

single = model.predict(df)
probability = model.predict_proba(df)[:, 1]
probability = probability*100

if single == 1:
    op1 = "This Customer is likely to be Churned!"
    op2 = f"Confidence level is {np.round(probability[0], 2)}"
else:
    op1 = "This Customer is likely to be Continue!"
    op2 = f"Confidence level is {np.round(probability[0], 2)}"

st.write(op1)
st.write(op2)