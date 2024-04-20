import numpy as np
import pandas as pd
import streamlit as st
import openai
import os
import requests
import joblib
from langchain import llms
from langchain.llms import OpenAI
import spacy


# Calling the pretrained model
MODEL_URL = 'https://github.com/dkamp007/Capstone/blob/main/Logistic2_Smote_Model.joblib'

# Function to download the model file
def download_model(model_url, save_path):
    try:
        response = requests.get(model_url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully.")
    except Exception as e:
        print("Error downloading model:", e)

# Path to save the downloaded model file
model_path = 'Logistic2_Smote_Model.joblib'

# Checking if the model file already exists
if not os.path.exists(model_path):
    # Download the model file
    download_model(MODEL_URL, model_path)

# Load the model
model = joblib.load(model_path)

# openai api key
api = st.secrets['apikey']
os.environ['OPENAI_API_KEY'] = api


# Label encoding of the categorical features as defined in the pretrained model
category_encoding_map = {
  "City Tier": {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2},
    
  "Property Insurance": {"Insured": 0, "Not Insured": 1},
    
  "Salary Type": {"Non-Salaried": 0, "Salaried": 1},
    
  "Loan Sub-Type": {"Builder Ready": 0, "Builder Under-Construction": 1, "Others": 2, "Resale Ready": 3, "Self Construction": 4},

  "Salaried/Self-Employed": {"Others": 0, "Salaried": 1, "Self Employed": 2},

  "Form 16": {"Not-Submitted": 0, "Submitted": 1},

  "Credit Bureau Enquiries": {"More than 0 & Less than equals 2": 0, "More than 10": 1,
                              "More than 2 & Less than equals 4": 2, "More than 4 & Less than equals 10": 3},

  "Delinquent Tradelines": {"Have Delinquents": 0, "No Delinquents": 1},

  "Minimum Vintage Tradelines": {"More than 0 & Less than equals 1": 0, "More than 1 & Less than equals 2": 1,
                                 "More than 2 & Less than equals 3": 2, "More than 3 & Less than equals 5": 3,
                                 "More than 5": 4},

  "Loan Accounts": {"More than 0 & Less than equals 1": 0, "More than 1 & Less than equals 2": 1,
                    "More than 2 & Less than equals 4": 2, "More than 4 & Less than equals 8": 3,
                    "More than 9": 4},

  "Active Loans": {"Zero Active": 0, "One Active": 1, "Two Active": 2,
                   "More than 2 & Less than equals 4 Active": 3, "More than 4": 4}  
}



img = 'https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'


# Title
st.title("Hi, I am a :red[Loan] :blue[Approval] :orange[Predictor] Bot :robot_face:")

st.divider()

# Creating a space for better visual separation
st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)

st.markdown(f'<div style="display: flex; justify-content: center;"><img src="{img}" width="780"></div>', unsafe_allow_html=True)

# Creating a space for better visual separation
st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)

#st.divider()

st.sidebar.header("Welcome! This is a friendly app")
st.sidebar.markdown('''This app is going to help you predict whether your loan application will be **approved/rejected.**
                        Feel free to explore and interact.''')



# Function to extract features from user input using NLP (for numerical features)
def extract_from_text(user_input):
    nlp = spacy.load('en_core_web_sm')
    docs = nlp(user_input)

    extracted_features = {"Income": None, "Tenure": None, "LTV": None, "EMI": None, "Loan Amount": None, "property value": None}

    for token in docs:
        if not token.is_punct:
            if token.like_num:
                context_window = [token.text.lower() for token in docs[token.i - 6: token.i + 3]]

                if any(keyword in context_window for keyword in ['income', 'salary']):
                    extracted_features["Income"] = float(token.text)
                elif any(keyword in context_window for keyword in ['tenure', 'term', 'years']):
                    extracted_features["Tenure"] = float(token.text)
                elif any(keyword in context_window for keyword in ['property', 'worth', 'assets', 'value', 'valued',
                                                                   'collateral']):
                    extracted_features["property value"] = float(token.text)
                elif any(keyword in context_window for keyword in ['loan', 'amount', 'loan-amount']):
                    extracted_features["Loan Amount"] = float(token.text)
                elif any(keyword in context_window for keyword in ['emi', 'monthly', 'payments', 'installment', 'installments']):
                    extracted_features["EMI"] = float(token.text)

    # Calculate LTV ratio if both 'Loan Amount' and 'property value' are available
    loan = extracted_features.get("Loan Amount")
    ppty = extracted_features.get("property value")

    if loan is not None and ppty is not None and ppty != 0:
        ltv_ratio = loan / ppty
        extracted_features["LTV"] = round(ltv_ratio, 4)

    return extracted_features




# Function to perform Box-Cox transformation
def income_transform(value, lambda_value):
    # Adding a small constant to handle zero values
    #value = value + 1e-5
    transformed_value = (value**lambda_value - 1) / lambda_value
    return transformed_value



# Numerical features - text input

st.subheader('Please Enter Details of :blue[Income, Tenure], :orange[Loan Amount, Property Value] & EMI', divider='rainbow')

#user_input_numericals = st.text_input('Insert', help='Enter your details in plain English!')
user_input_numericals = st.text_area('Insert', help='Enter your details in plain English!')

numericals_from_text = extract_from_text(user_input_numericals)

# Metric display

inc = numericals_from_text.get('Income')

tenure = numericals_from_text.get('Tenure')

ltv = numericals_from_text.get('LTV')

emi = numericals_from_text.get('EMI')


# Checking if any numerical feature is missing
missing_features = [key for key, value in numericals_from_text.items() if value is None]

if missing_features:
    st.warning(f"You haven't provided the following details: {', '.join(missing_features)}")


col1, col2, col3, col4 = st.columns(4)
col1.metric("**Income**", value=inc)
col2.metric("**Tenure**", value=tenure)
col3.metric("**LTV Ratio**", value=ltv)
col4.metric('**EMI**', value=emi)

# Storing the original input for llm prompt
numericals_from_text_original = {"Income": inc, "Tenure": tenure, "LTV": ltv, "EMI": emi}

#st.divider()


# Applying the transformation on the income value extracted from the input
income_value = numericals_from_text["Income"]
lambda_value = -0.5448345091305346
# Check if income_value is not None before performing the calculation
if income_value is not None:          
    # Perform the Box-Cox transformation on the income feature
    income_for_pred = income_transform(income_value, lambda_value)
    # Update the user_features dictionary with the new income value    
    numericals_from_text["Income"] = income_for_pred



# Calculating the amount of loan eligible for a burrower
def loan_calculator(p,r,t):
    r = r/(12 * 100) # monthly interest rate
    t = t * 12 # time in month
    loan_amt = p/((r * pow(1+r, t))/(pow(1+r, t) -1)) #
    return loan_amt


# Streamlit app
st.sidebar.header("Loan Calculator")
st.sidebar.markdown('''This is a simple calculator for you to check the maximum amount of loan you are eligible to get.
                            Feel free to use it.''')


amt1 = st.sidebar.number_input("Monthly Income", min_value=0, value=None)
amt2 = st.sidebar.number_input("Monthly EMI", min_value=0, value=None)
interest_rate = st.sidebar.slider("Interest Rate (%)", min_value=0.5, max_value=15.0, step=0.1, value=5.0)
tenure_side = st.sidebar.slider("Loan Tenure (Years)", min_value=1, max_value=40, value=12)

# User input widgets
if amt1 is not None and amt2 is not None:    
    amt_left = amt1 - amt2
    
    # Applying the above function to create a new feature of Loan_eligible
    Loan_Eligible = loan_calculator(amt_left, interest_rate, tenure_side)

    if Loan_Eligible < 0:
        st.sidebar.write('**Sorry!** It appears that your **EMI** is **greater** than your **Income**.')

    else:
        # Display the loan amount
        st.sidebar.metric('Loan Eligible', round(Loan_Eligible, 2))
    



# Dividing into two columns
st.subheader('Select The :blue[Appropriate] :orange[Options]', divider='rainbow')
col1, col2 = st.columns(2)

# Categorical features - dropdown menus
with col1:
    
    city_tier_options = ["Tier 1", "Tier 2", "Tier 3"]
    city_tier = st.selectbox("**City Tier**", city_tier_options, index=None,
                             placeholder="Select your option")#, key=None)
    st.write('You selected:', city_tier)

    property_insurance_options = ["Insured", "Not Insured"]
    property_insurance = st.selectbox("**Property Insurance**", property_insurance_options, index=None,
                                placeholder="Select your option")#, key="property_insurance")
    st.write('You selected:', property_insurance)

    salary_type_options = ["Salaried", "Non-Salaried"]
    salary_type = st.selectbox("**Salary Type**", salary_type_options, index=None,
                                placeholder="Select your option")#, key="salary_type")
    st.write('You selected:', salary_type)

    loan_type_options = ["Builder Ready", "Builder Under-Construction", "Resale Ready", "Self Construction", "Others"]
    loan_type = st.selectbox("**Loan Sub-Type**", loan_type_options, index=None,
                                placeholder="Select your option")#, key="loan_type")
    st.write('You selected:', loan_type)

    sal_self_options = ["Salaried", "Self Employed", "Others"]
    Sal_Self = st.selectbox("**Salaried/Self-Employed**", sal_self_options, index=None,
                                placeholder="Select your option")#, key="Sal_Self")
    st.write('You selected:', Sal_Self)


with col2:
     
    form16_options = ["Submitted", "Not-Submitted"]
    form16 = st.selectbox("**Form 16**", form16_options, index=None,
                             placeholder="Select your option")#, key="form16")
    st.write('You selected:', form16)

    enq_options = ["More than 0 & Less than equals 2", "More than 2 & Less than equals 4", "More than 4 & Less than equals 10", "More than 10"]
    enq = st.selectbox("**Credit Bureau Enquiries**", enq_options, index=None,
                                placeholder="Select your option")#, key="enq")
    st.write('You selected:', enq)

    dq_tl_options = ["Have Delinquents", "No Delinquents"]
    dq = st.selectbox("**Delinquent Tradelines**", dq_tl_options, index=None,
                                placeholder="Select your option")#, key="dq")
    st.write('You selected:', dq)

    min_age_tl_options = ["More than 0 & Less than equals 1", "More than 1 & Less than equals 2", "More than 2 & Less than equals 3", "More than 3 & Less than equals 5", "More than 5"]
    min_age_tl = st.selectbox("**Minimum Vintage Tradelines**" , min_age_tl_options, index=None,
                                placeholder="Select your option")#, key="min_age_tl")
    st.write('You selected:', min_age_tl)

    loan_acc_options = ["More than 0 & Less than equals 1", "More than 1 & Less than equals 2", "More than 2 & Less than equals 4", "More than 4 & Less than equals 8", "More than 9"]
    loan_acc = st.selectbox("**Loan Accounts**", loan_acc_options, index=None,
                                placeholder="Select your option")#, key="loan_acc")
    st.write('You selected:', loan_acc)

    
open_tl_options = ["Zero Active", "One Active", "Two Active", "More than 2 & Less than equals 4 Active", "More than 4"]
open_tl = st.selectbox("**Number of Active Loans**", open_tl_options, index=None,
                                placeholder="Select your option")#, key="open_tl")
st.write('You selected:', open_tl)

st.divider()


# Creating a dictionary for all the details given by the user
user_features_extracted = {
        "Salary Type": salary_type, "Salaried/Self-Employed": Sal_Self, "Loan Sub-Type":loan_type, 
        "Form 16": form16, "Property Insurance": property_insurance, "City Tier": city_tier,
        "Credit Bureau Enquiries": enq, "Delinquent Tradelines": dq, "Minimum Vintage Tradelines": min_age_tl,
        "Loan Accounts": loan_acc, "Active Loans": open_tl,
      "LTV": numericals_from_text.get("LTV"), "Tenure": numericals_from_text.get("Tenure"),
      "EMI": numericals_from_text.get("EMI"), "Income": numericals_from_text.get("Income")
    }




# Creating a dictionary with the original inputs by the user for llm prompt
user_features_extracted_original = {
      "Salary Type": salary_type, "Salaried/Self-Employed": Sal_Self, "Loan Sub-Type":loan_type, 
        "Form 16": form16, "Property Insurance": property_insurance, "City Tier": city_tier,
        "Credit Bureau Enquiries": enq, "Delinquent Tradelines": dq, "Minimum Vintage Tradelines": min_age_tl,
        "Loan Accounts": loan_acc, "Active Loans": open_tl,
      "LTV": numericals_from_text_original.get("LTV"), "Tenure": numericals_from_text_original.get("Tenure"),
      "EMI": numericals_from_text_original.get("EMI"), "Income": numericals_from_text_original.get("Income")
    }



# Mapping categorical features to label encoded values
def encode_categorical_features(selected_options):
    encoded_values = {}
    for feature_name, selected_option in selected_options.items():
        # Check if selected_option is not None
        if selected_option is not None:
            # Using category_encoding_map to get numerical label
            encoding_label = category_encoding_map[feature_name][selected_option]
            encoded_values[feature_name] = encoding_label
    return encoded_values



# Creating a dictionary to store selected options
selected_options = {
    "City Tier": city_tier,
    "Property Insurance": property_insurance,
    "Salary Type": salary_type,
    "Loan Sub-Type": loan_type,
    "Salaried/Self-Employed": Sal_Self,
    "Form 16": form16,
    "Credit Bureau Enquiries": enq,
    "Delinquent Tradelines": dq,
    "Minimum Vintage Tradelines": min_age_tl,
    "Loan Accounts": loan_acc,
    "Active Loans": open_tl
}



encoded_values = encode_categorical_features(selected_options)
# Updating encoded values into user_features_extracted
user_features_extracted.update(encoded_values)


# Output from here
st.header('The Output', divider='rainbow')


# Button to trigger prediction
if st.button("Predict"):

    # DataFrame with feature names as columns and a single row with corresponding values for llm prompt
    feature_names1 = list(user_features_extracted_original.keys())
    feature_values1 = list(user_features_extracted_original.values())
    extracted_features_df_original = pd.DataFrame([feature_values1], columns=feature_names1)

    # Extract only the features to include in the explanation
    selected_features = extracted_features_df_original[['City Tier', 'Property Insurance', 'Salary Type', 'Income',
                                                        'EMI', 'Tenure', 'Delinquent Tradelines']]

    # Converting the selected features to a string for inclusion in the prompt
    selected_features_str = selected_features.to_string(index=False, header=False)

    
    # Assuming user_features_extracted is the dictionary
    feature_names = list(user_features_extracted.keys())
    feature_values = list(user_features_extracted.values())
  
    # DataFrame with feature names as columns and a single row with corresponding values for prediction
    extracted_features_df = pd.DataFrame([feature_values], columns=feature_names)

    # Initializing the llm
    llm = OpenAI(temperature=0.1)

    # Using extracted_features_df for prediction
    prediction = model.predict(extracted_features_df)[0]


    # Convert prediction into natural language
    prediction_text = "approved" if prediction == 1 else "rejected"
    prediction_text1 = "Congratulations! It is approved." if prediction == 1 else "Sorry! It is rejected."

    st.write(prediction_text1)

    # Displaying the dataframe for the user details
    st.write('Your details are as follows:')
    st.dataframe(extracted_features_df_original)


    # Generate explanation using OpenAI directly

#    prompt = f"Explain the reasons behind the prediction ({prediction_text}) for this user's loan eligibility, considering the following features used in the prediction using machine learning:\n- City Tier: {extracted_features_df_original['City Tier'].values[0]}\n- Property Insurance: {extracted_features_df_original['Property Insurance'].values[0]}\n- Salary Type: {extracted_features_df_original['Salary Type'].values[0]}\n- Loan Sub-Type: {extracted_features_df_original['Loan Sub-Type'].values[0]}\n- Salaried/Self-Employed: {extracted_features_df_original['Salaried/Self-Employed'].values[0]}\n- Form 16: {extracted_features_df_original['Form 16'].values[0]}\n- Credit Bureau Enquiries: {extracted_features_df_original['Credit Bureau Enquiries'].values[0]}\n- Delinquent Tradelines: {extracted_features_df_original['Delinquent Tradelines'].values[0]}\n- Minimum Vintage Tradelines: {extracted_features_df_original['Minimum Vintage Tradelines'].values[0]}\n- Loan Accounts: {extracted_features_df_original['Loan Accounts'].values[0]}\n- Active Loans: {extracted_features_df_original['Active Loans'].values[0]}\n- LTV: {extracted_features_df_original['LTV'].values[0]}\n- Tenure: {extracted_features_df_original['Tenure'].values[0]}\n- EMI: {extracted_features_df_original['EMI'].values[0]}\n- Income: {extracted_features_df_original['Income'].values[0]}"


    # Generating the prompt with bullet points
    bullet_prompt = f"Explain the reasons behind the prediction ({prediction_text}) for this user's loan eligibility, considering the following features used in the prediction using machine learning:\n"
    bullet_prompt += "- City Tier: {}\n".format(extracted_features_df_original['City Tier'].values[0])
    bullet_prompt += "- Property Insurance: {}\n".format(extracted_features_df_original['Property Insurance'].values[0])
    bullet_prompt += "- Salary Type: {}\n".format(extracted_features_df_original['Salary Type'].values[0])
    bullet_prompt += "- Loan Sub-Type: {}\n".format(extracted_features_df_original['Loan Sub-Type'].values[0])
    bullet_prompt += "- Salaried/Self-Employed: {}\n".format(extracted_features_df_original['Salaried/Self-Employed'].values[0])
    bullet_prompt += "- Form 16: {}\n".format(extracted_features_df_original['Form 16'].values[0])
    bullet_prompt += "- Credit Bureau Enquiries: {}\n".format(extracted_features_df_original['Credit Bureau Enquiries'].values[0])
    bullet_prompt += "- Delinquent Tradelines: {}\n".format(extracted_features_df_original['Delinquent Tradelines'].values[0])
    bullet_prompt += "- Minimum Vintage Tradelines: {}\n".format(extracted_features_df_original['Minimum Vintage Tradelines'].values[0])
    bullet_prompt += "- Loan Accounts: {}\n".format(extracted_features_df_original['Loan Accounts'].values[0])
    bullet_prompt += "- Active Loans: {}\n".format(extracted_features_df_original['Active Loans'].values[0])
    bullet_prompt += "- LTV: {}\n".format(extracted_features_df_original['LTV'].values[0])
    bullet_prompt += "- Tenure: {}\n".format(extracted_features_df_original['Tenure'].values[0])
    bullet_prompt += "- EMI: {}\n".format(extracted_features_df_original['EMI'].values[0])
    bullet_prompt += "- Income: {}\n".format(extracted_features_df_original['Income'].values[0])

    
    #response = llm(prompt, max_tokens=2000)
    response = llm(bullet_prompt, max_tokens=2000)

    
    # Check if the response is not empty
    if response:
        explanation = response.strip()
        st.write("Explanation:\n", explanation)
    else:
        st.write("LLM response is empty.")
