import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained XGBoost model
with open('model/credit_risk_xgb.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the web app
st.title('Credit Risk Prediction')

# Define the full set of features that the model expects
expected_features = [
    'last_fico_range_high', 'last_fico_range_low', 'recoveries', 'collection_recovery_fee',
    'last_pymnt_amnt', 'grade', 'int_rate', 'out_prncp', 'out_prncp_inv', 'fico_range_low',
    'fico_range_high', 'acc_open_past_24mths', 'dti', 'num_tl_op_past_12m', 'mort_acc',
    'bc_open_to_buy', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'avg_cur_bal'
]

# Option to input manually or upload CSV using radio buttons
input_option = st.radio("Choose input method", ["Input manually", "Upload CSV"])

def add_interaction_features(df):
    # Create interaction term 'last_fico_range_high*last_fico_range_low'
    df['last_fico_range_high*last_fico_range_low'] = df['last_fico_range_high'] * df['last_fico_range_low']
    return df

if input_option == "Input manually":
    st.write("Please enter the following details:")

    # Input form for user to enter the required features
    last_fico_range_high = st.number_input('Last FICO Range High', value=600.0)
    last_fico_range_low = st.number_input('Last FICO Range Low', value=600.0)
    recoveries = st.number_input('Recoveries', value=0.0)
    collection_recovery_fee = st.number_input('Collection Recovery Fee', value=0.0)
    last_pymnt_amnt = st.number_input('Last Payment Amount', value=1500.0)
    grade = st.selectbox('Grade', options=['A', 'B', 'C', 'D', 'E', 'F', 'G'], index=0)
    int_rate = st.number_input('Interest Rate (%)', value=10.0)
    out_prncp = st.number_input('Outstanding Principal', value=1000.0)
    out_prncp_inv = st.number_input('Outstanding Principal (Investor)', value=1000.0)
    fico_range_low = st.number_input('FICO Range Low', value=600.0)
    fico_range_high = st.number_input('FICO Range High', value=600.0)
    acc_open_past_24mths = st.number_input('Accounts Open Past 24 Months', value=5)
    dti = st.number_input('Debt-to-Income Ratio', value=20.0)
    num_tl_op_past_12m = st.number_input('Number of Trades Opened Past 12 Months', value=2)
    mort_acc = st.number_input('Mortgage Accounts', value=1)
    bc_open_to_buy = st.number_input('Bankcard Open to Buy', value=5000.0)
    loan_amnt = st.number_input('Loan Amount', value=10000.0)
    funded_amnt = st.number_input('Funded Amount', value=10000.0)
    funded_amnt_inv = st.number_input('Funded Amount (Investor)', value=10000.0)
    avg_cur_bal = st.number_input('Average Current Balance', value=2000.0)

    # Convert 'grade' to numerical format
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    grade_value = grade_map[grade]

    # Create a DataFrame from the user input
    user_input = pd.DataFrame({
        'last_fico_range_high': [last_fico_range_high],
        'last_fico_range_low': [last_fico_range_low],
        'recoveries': [recoveries],
        'collection_recovery_fee': [collection_recovery_fee],
        'last_pymnt_amnt': [last_pymnt_amnt],
        'grade': [grade_value],
        'int_rate': [int_rate],
        'out_prncp': [out_prncp],
        'out_prncp_inv': [out_prncp_inv],
        'fico_range_low': [fico_range_low],
        'fico_range_high': [fico_range_high],
        'acc_open_past_24mths': [acc_open_past_24mths],
        'dti': [dti],
        'num_tl_op_past_12m': [num_tl_op_past_12m],
        'mort_acc': [mort_acc],
        'bc_open_to_buy': [bc_open_to_buy],
        'loan_amnt': [loan_amnt],
        'funded_amnt': [funded_amnt],
        'funded_amnt_inv': [funded_amnt_inv],
        'avg_cur_bal': [avg_cur_bal],
    })

    # Add interaction feature
    user_input = add_interaction_features(user_input)

    # Ensure the input has the same features as the model expects
    for feature in expected_features:
        if feature not in user_input.columns:
            user_input[feature] = 0  # Add missing feature with default value 0

    # Reorder columns to match the model's expected order
    user_input = user_input[expected_features]

    # Button to make the prediction
    if st.button('Predict'):
        # Make prediction using the model
        prediction = model.predict(user_input)

        # Display the prediction result
        if prediction[0] > 0.5:
            st.write("Prediction: High Risk of Credit Default")
        else:
            st.write("Prediction: Low Risk of Credit Default")

else:
    # Option to upload CSV file
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Check the columns of the uploaded file
        st.write("Uploaded Data:")
        st.write(df)

        # Ensure only the selected features are present
        df = df[expected_features]

        # Add interaction feature
        df = add_interaction_features(df)

        # Button to make prediction
        if st.button('Predict'):
            prediction = model.predict(df)

            # Show predictions
            prediction_results = ["High Risk" if p > 0.5 else "Low Risk" for p in prediction]

            # Show prediction results
            st.write("Prediction Results:")
            st.write(pd.DataFrame({
                'Prediction': prediction_results
            }))
