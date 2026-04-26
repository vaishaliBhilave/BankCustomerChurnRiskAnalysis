# ==========================================================
# Importing necessary libraries
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import shap

# ==========================================================
# Set Page Configuration   
# ==========================================================
st.set_page_config(
    page_title="Bank Customer Churn Intelligence System",
    page_icon="https://www.ecb.europa.eu/shared/img/logo/logo_only.svg",
    layout="wide"
)

col1, col2 = st.columns([1,4])

with col1:
    st.image("https://www.ecb.europa.eu/shared/img/logo/logo_only.svg", width=120)

with col2:
    st.title("Bank Customer Churn Intelligence System")
    st.write("This application predicts the probability of bank customer churning based on various features. It also provides insights into the factors contributing to customer churn.")
# ==========================================================
# Load the pre-trained model and data
# ==========================================================
@st.cache_resource
def load_model():
    model = joblib.load("churn_model.pkl")
    return model

model = load_model()

# Side Navigation Bar for user input
st.sidebar.header("Customer Information")
creditScore = st.sidebar.slider("Credit Score", 300, 850, 650)  
geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
gender  = st.sidebar.selectbox("Gender",["Male","Female"])
age = st.sidebar.slider("Age", 18, 100, 30)
year = st.sidebar.slider("Year", 2010, 2024, 2020)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3)  
accBalance = st.sidebar.number_input("Account Balance", min_value=0.0, max_value=300000.0, value=60000.0, step=1000.0 )
noOfProducts = st.sidebar.slider("Number of Products", 1, 4, 2)
hasCrCard = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
isActiveMember = st.sidebar.selectbox("Is Active Member?", ["Yes", "No"])
estimatedSalary = st.sidebar.number_input("Estimated Salary", min_value=10000.0, max_value=200000.0, value=50000.0, step=1000.0)  

hasCrCard = 1 if hasCrCard == "Yes" else 0
isActiveMember = 1 if isActiveMember == "Yes" else 0

# ==========================================================
# Feature Engineering 
# ========================================================= 
balance_salary_ratio = accBalance/(estimatedSalary+1)

product_density = noOfProducts/(tenure+1)

engagement_score = isActiveMember*noOfProducts

age_tenure_ratio = age/(tenure+1)

# ==========================================================
# Prepare the input data for prediction
input_data = pd.DataFrame({
    "CreditScore": [creditScore],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Year": [year],   
    "Balance": [accBalance],
    "NumOfProducts": [noOfProducts],
    "HasCrCard": [hasCrCard],
    "IsActiveMember": [isActiveMember],
    "EstimatedSalary": [estimatedSalary],   
    "BalanceSalaryRatio": [balance_salary_ratio],
    "ProductDensity": [product_density],
    "EngagementScore": [engagement_score],
    "AgeTenureRatio": [age_tenure_ratio]
})  

# ==========================================================
# Make prediction using the loaded model
prediction_prob = model.predict_proba(input_data)[0][1]  # Probability of churning
threshold = 0.5
prediction = int(prediction_prob >= threshold)
st.subheader("Churn Prediction")    
st.write(f"The probability of the customer churning is: {prediction_prob:.2%}")

# ==========================================================
# Display Dashboard Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Risk Assessment","Churn Probability Analysis","Feature Importance", "What-if scenario simulator"])
with tab1:
    st.subheader("Churning Risk Assessment")
    col1,col2 = st.columns(2)

    with col1:
        st.metric("Churning Risk Assessment",f"{round(prediction_prob*100,2)} %")
        if prediction_prob <0.30:
            risk="Low Risk"
        elif prediction_prob<0.65:
            risk="Medium Risk"
        else:
            risk="High Risk"
        st.metric("Churning Risk Level",risk)    
       
    with col2:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction_prob*100,
            title = {'text': "Churn Probability Risk Gauge"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps' : [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 65], 'color': "yellow"},
                    {'range': [65, 100], 'color': "red"}
                ],
            }
        ))
        st.plotly_chart(fig)    

with tab2:
    st.subheader("Churn Probability Analysis")
    retention_rate = 1 - prediction_prob
    fig , ax = plt.subplots()
    ax.bar(["Retention Rate", "Churn Probability"], [retention_rate, prediction_prob], color=["green", "red"])   
    ax.set_ylim(0, 1)
    ax.set_title("Customer Retention vs Churn Probability") 
    st.pyplot(fig)

with tab3:
    st.subheader("Feature Importance")
    model_tree = model.named_steps["model"]
    # Extract model from pipeline
    gb_model = model.named_steps["model"]

    # Get feature importance
    feature_importance = gb_model.feature_importances_

    # Get feature names after preprocessing
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    # Create dataframe
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))

    # ==============================
    # SHAP EXPLAINABILITY
    # ==============================
    st.subheader("🧠 SHAP Explanation")

    try:
        preprocessor = model.named_steps['preprocessor']
        # Transform input
        X_transformed = preprocessor.transform(input_data)
        # ✅ Convert to DataFrame (IMPORTANT FIX)
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

        # SHAP explainer

        explainer = shap.TreeExplainer(gb_model)
        print("🧠 SHAP Explanation",explainer )
        shap_values = explainer(X_transformed_df)
        print("SHAP values calculated successfully", shap_values)

        fig2, ax2 = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig2)

    except:
        st.warning("SHAP visualization not available")

with tab4:
    st.subheader("What-if Scenario Simulator")
    change_products = st.slider("Change Number of Products", 1, 4, noOfProducts)
    change_active = st.selectbox("Change Active Member Status", ["Yes", "No"])
    change_balance = st.number_input("Change Account Balance", min_value=0.0,   max_value=300000.0, value=accBalance, step=1000.0)
    change_salary = st.number_input("Change Estimated Salary", min_value=10000.0, max_value=200000.0, value=estimatedSalary, step=1000.0)
    change_balance_salary_ratio = change_balance/(change_salary+1)
    change_product_density = change_products/(tenure+1)
    change_engagement_score = (1 if change_active == "Yes" else 0)*change_products
    what_if_data = pd.DataFrame({
        "CreditScore": [creditScore],
        "Geography": [geography], 
        "Gender": [gender], 
        "Age": [age],
        "Tenure": [tenure],
        "Year": [year],
        "Balance": [change_balance],
        "NumOfProducts": [change_products],
        "HasCrCard": [1 if hasCrCard == "Yes" else 0],
        "IsActiveMember": [1 if change_active == "Yes" else 0],
        "EstimatedSalary": [change_salary],
        "BalanceSalaryRatio": [change_balance_salary_ratio],
        "ProductDensity": [change_product_density],
        "EngagementScore": [change_engagement_score],
        "AgeTenureRatio": [age_tenure_ratio]
    })
    what_if_prob = model.predict_proba(what_if_data)[0][1]
    st.metric("New Churn Probability", f"{what_if_prob:.2%}")
    probChange = (what_if_prob - prediction_prob) *100  
    st.metric("Change in Churn Probability", f"{round(probChange,2)} %")
    
    #==============================
    # WHAT-IF ANALYSIS
    # ==============================
    st.subheader("🔄 What-if Scenario (Products vs Churn)")

    product_range = list(range(1, 5))
    prob_list = []

    for p in product_range:
        what_if_data["NumOfProducts"] = p
        prob = model.predict_proba(what_if_data)[0][1]
        prob_list.append(prob)

    # Plotting the results with Matplotlib and figure size adjustment for better visibility 
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(product_range, prob_list, marker='o')
    ax3.set_xlabel("Number of Products")
    ax3.set_ylabel("Churn Probability")
    st.pyplot(fig3)


    try:
        preprocessor = model.named_steps['preprocessor']
        X_transformed_what_if = preprocessor.transform(what_if_data)
        X_transformed_what_if_df = pd.DataFrame(X_transformed_what_if, columns=feature_names)
        explainer = shap.TreeExplainer(gb_model)
        shap_values_what_if = explainer(X_transformed_what_if_df)
        fig4, ax4 = plt.subplots()
        shap.plots.waterfall(shap_values_what_if[0], show=False)
        st.pyplot(fig4)
    except:
        st.warning("SHAP visualization not available")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.write("© 2024 Bank Customer Churn Intelligence System. All rights reserved.")    
st.write("Developed by Vaishali.")


    

    


