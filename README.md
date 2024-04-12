# Customer Churn-Prediction

### Telecom Customer Churn Prediction

Customer churn refers to the phenomenon where customers or subscribers stop doing business with a company or stop using its services. 
High churn rates can indicate dissatisfaction among customers or issues with the product or service offered by the company.
Hence identifying high risk customers who are likely to leave is a cruicial step to focus on these segment of customers to try and reatain them by providing certain incentives or improving certain aspects that may lead to retention.

### Deployment
Deployed as a streamlit web app - https://telcom-customer-churn-prediction.streamlit.app/

![image](https://github.com/RohitMacherla3/customer-churn-prediction/assets/89356811/9708847e-b57a-4ee1-940f-9ba4358697e5)

### Objective
- This can be achieved by building a classification model to predict if a customer is likely to leave the company based on certain user features.
- Explore different models ranging from simple logistic regression to decision trees and neural networks to compare and identify best performing and most suitable model.

### Dataset

The Dataset is obtain from [IBM sample data](https://www.ibm.com/docs/en/cognos-analytics/11.1.0?topic=samples-telco-customer-churn) which is also available on [Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset).
The Dataset has columns consisting of the following information:
- Customer Demographics information - gender, age, partner, dependents, city, zip code, latitude and longitude
- Customer Account info - features related to contract, payment method, paperless billing, monthly charges, and total charges
- Customer Services info - features related to phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Churn Info - If the user stopped using the product in the past month

### EDA

1. Correlation of different feature with the predictor
![image](https://github.com/RohitMacherla3/customer-churn-prediction/assets/89356811/0cd912f4-f174-4a84-9274-d47fc4dc992a)

2. Feature Importance from XGBoost
![image](https://github.com/RohitMacherla3/customer-churn-prediction/assets/89356811/e8c8b371-3b1b-434a-82a3-fb348549975d)


### Models Used
- Logistic Regression
- Gaussian Naive Bayes
- Random Forest
- Gradient Boot
- XGBoost
- Kernel SVM

Note -  Hyperparameter tuning was performed for Random Forest and XGBosst to get optimal performance.

### Model Evaluation Comparisions

- Initial Comparisions
 ![image](https://github.com/RohitMacherla3/customer-churn-prediction/assets/89356811/ca37130e-c7fe-4f7e-ae2d-7b758afac0a5)


- After SMOTEEN (over-sampling to deal with class imbalances)
![image](https://github.com/RohitMacherla3/customer-churn-prediction/assets/89356811/68d09ed6-5cac-4e62-8058-ac2af9c7a70b)

- Best Performing Model (XGBoost)
![image](https://github.com/RohitMacherla3/customer-churn-prediction/assets/89356811/cc2a6150-dd49-45e0-a0ec-8f2522be0979)
