import streamlit as st
import pandas as pd
import shap
import xgboost as xgb
import numpy as np
import joblib

# Load the trained model
# model = xgb.XGBClassifier()
model = joblib.load("model/xgb_model.pkl")  # Path to your model file

# Feature names
feature_names = [
    "dti", "installment", "fico_range_low", "loan_amnt", "annual_inc",
    "int_rate", "fico_range_high", "emp_length", "delinq_2yrs", "grade"
]

# Streamlit app
st.title("Credit Risk Prediction with SHAP Explanation")
st.markdown("This app predicts credit risk based on input values and provides SHAP explanations.")

# Input collection
st.subheader("Enter Feature Values for Prediction")
user_inputs = {}
for feature in feature_names:
    user_inputs[feature] = st.number_input(f"Enter value for {feature}:", value=0.0)

# Option to upload JSON file
st.subheader("OR Upload a JSON File")
uploaded_file = st.file_uploader("Upload a JSON file with feature values", type=["json"])

if uploaded_file:
    user_inputs = pd.read_json(uploaded_file, typ='series').to_dict()

# Prediction button
if st.button("Predict"):
    # Convert user inputs to DataFrame
    input_data = pd.DataFrame([user_inputs])

    # Prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Display prediction results
    st.subheader("Prediction Result")
    if prediction == 1:
        st.write("The model predicts that this case is **high risk**.")
    else:
        st.write("The model predicts that this case is **low risk**.")
    
    st.subheader("Prediction Probability")
    st.write(f"Low Risk: {prediction_proba[0]:.2f}")
    st.write(f"High Risk: {prediction_proba[1]:.2f}")

    # SHAP Explanation
    st.subheader("SHAP Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # SHAP summary plot
    st.write("Feature Impact (Bar Plot)")
    shap.summary_plot(shap_values, input_data, feature_names=feature_names, show=False)
    st.pyplot()  # Render SHAP plot

    # SHAP waterfall plot for individual prediction
    st.write("Detailed Explanation (Waterfall Plot)")
    shap.waterfall_plot(
        shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, 
                         data=input_data.iloc[0].values, feature_names=feature_names)
    )
    st.pyplot()
