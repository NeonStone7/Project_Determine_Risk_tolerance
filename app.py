import numpy as np
import pandas as pd
import streamlit as st
import zipfile
import io
import joblib  
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import set_config
set_config(transform_output = 'pandas')

# Function to load the model from a zipped file
def load_model_from_zip(zip_file_path, model_filename):
    
    # Read the zipped file as bytes
    with open(zip_file_path, 'rb') as f:
        zip_data = io.BytesIO(f.read())

    # Open the zip archive
    with zipfile.ZipFile(zip_data) as zip_ref:
        # Load the model file from the zip archive
        with zip_ref.open(model_filename) as model_file:
            # Assuming your model is serialized using joblib
            model = joblib.load(model_file)
    return model

# Path to your zipped model file
zip_file_path = 'modelPkl.zip'

# Name of the model file inside the zip file (e.g., 'model.pkl')
model_filename = 'modelPkl.pkl'

# Load the model from the zipped file
try:
    model = load_model_from_zip(zip_file_path, model_filename)
    st.write("Model loaded successfully!")
    
    # Now you can use the loaded model for predictions or other tasks
except Exception as e:
    st.error(f"Error loading the model: {e}")

loaded_preprocessor = joblib.load('preprocessor_pkl.pkl')

selected_columns = ['AGE',
 'Education_years',
 'Amount_on_homefood',
 'Amount_on_takeout',
 'INCOME',
 'AGE^2',
 'AGE Education_years',
 'AGE Amount_on_homefood',
 'AGE Amount_on_takeout',
 'AGE INCOME',
 'AGE Level_of_financial_literacy',
 'Education_years Amount_on_homefood',
 'Education_years Amount_on_takeout',
 'Education_years INCOME',
 'Education_years Level_of_financial_literacy',
 'Amount_on_homefood^2',
 'Amount_on_homefood Amount_on_takeout',
 'Amount_on_homefood INCOME',
 'Amount_on_homefood Level_of_financial_literacy',
 'Amount_on_takeout^2',
 'Amount_on_takeout INCOME',
 'Amount_on_takeout Level_of_financial_literacy',
 'INCOME^2',
 'INCOME Level_of_financial_literacy',
 'MARRIED',
 'Family_structure',
 'RACE',
 'Occupation',
 'Spending_to_income',
 'Late_payment',
 'Financial_Risk',
 'Use_emergency_savings',
 'Has_brokerage_account',
 'Traded_in_the_past_year',
 'Owns_business_assets',
 'Owns_non_financial_assets']

def main():
    
    st.title('Determine Risk Tolerance')
            
    st.markdown("""This model was trained on 10K samples\nTrain F1 Score: 97.5%\nTest F1 Score:71.8%""")
    
    st.markdown("""Disclaimer: This is a simple model. Please do not make any investment decisions based your result.""")

    st.markdown('Most important features for prediction')
    st.image('feature_importance.png')
    
    # input data
    
    sex = st.selectbox('What is your sex?', ['Female', 'Male'])
    age = st.number_input('How old are you?')
    married = st.selectbox('Are you married?', ['Married', 'Single'])
    Family_structure = st.selectbox('What is your family structure?',['Single abode','Married couple','Single Parent','Extended family','Other'])
    RACE = st.selectbox('What is your race?',['White/caucasian','Black/African-american', 'Hispanic/Latino','Asian','Others'])
    Occupation = st.selectbox('What is your occupation?',['Managerial/Professional','Technical/Sales/Services','Other','Not-working'])
    
    currency = st.selectbox('What is your currency?', ['USD','EUR', 'Naira', 'Others'])
     
    if currency == 'Others':
        
        st.markdown('Your currency is not available, please convert to dollar and input values in dollar')
        
    Amount_on_homefood = st.number_input('How much do you spend on home-cooked food annually?')
    
    Amount_on_takeout = st.number_input('How much do you spend on takeout food annually?')

    INCOME = st.number_input('What is your annual income?')
    
    educa = st.number_input('Number of years in education?')
    
    # convert to dollar, model was trained in dollar
    if currency == 'EUR':
        
        Amount_on_homefood = Amount_on_homefood/0.92
        Amount_on_takeout = Amount_on_takeout/0.92
        INCOME = INCOME/0.92
    
    elif currency == 'Naira':
        
        Amount_on_homefood = Amount_on_homefood/1469
        Amount_on_takeout = Amount_on_takeout/1469
        INCOME = INCOME/1469
  
    elif currency == 'Others':
        
        st.markdown('Your currency is not available, please convert to dollar and input values in dollar')
        

    Spending_to_income = st.selectbox('What is your spending to income category?',['Spending_exceeds_income','Spending_equals_income','Income_exceeds_spending'])

    Late_payment = st.selectbox('Any late loan payments in the last year?',['Yes','No'])

    Fear_loan_denial = st.selectbox('Do you fear being denied loans?',['Yes','No'])

    Use_emergency_savings = st.selectbox('Do you use your savings in a financial emergency or other options like cut-backs or loans?',['Yes','No'])

    Financial_Risk = st.selectbox('Are you willing to take financial risks?',['Yes','No'])

    Level_of_financial_literacy = st.selectbox('Level of financial literacy?',['0','1','2','3'])
    
    Level_of_financial_literacy = int(Level_of_financial_literacy)

    Has_brokerage_account = st.selectbox('Do you have a brokerage account?',['Yes','No'])

    Traded_in_the_past_year = st.selectbox('Have you traded in the past year?',['Yes','No'])

    Has_financial_assets = st.selectbox('Do you have financial assets?',['Yes','No'])

    Owns_home = st.selectbox('Do you own a home?',['Yes','No'])

    Owns_business_assets = st.selectbox('Do you own business assets?',['Yes','No'])

    Owns_non_financial_assets = st.selectbox('Do you own non-financial assets?',['Yes','No'])
    
    if st.button('Determine Risk tolerance'):
        
        data = pd.DataFrame({
        'Sex':sex, 'AGE':age, 'Education_years':educa, 'MARRIED':married, 'Family_structure':Family_structure,
        'RACE':RACE, 'Occupation':Occupation, 'Amount_on_homefood':Amount_on_homefood, 'Amount_on_takeout':Amount_on_takeout,
        'INCOME':INCOME,
        'Fear_loan_denial':Fear_loan_denial, 'Spending_to_income':Spending_to_income, 'Late_payment':Late_payment, 'Financial_Risk':Financial_Risk,
        'Use_emergency_savings':Use_emergency_savings, 'Level_of_financial_literacy':Level_of_financial_literacy,
        'Has_brokerage_account':Has_brokerage_account, 'Traded_in_the_past_year':Traded_in_the_past_year,
        'Has_financial_assets':Has_financial_assets, 'Owns_home':Owns_home, 'Owns_business_assets':Owns_business_assets,
        'Owns_non_financial_assets':Owns_non_financial_assets,

    }, index = [0])

        preprocessed_data = loaded_preprocessor.transform(data)

        preprocessed_data = preprocessed_data.loc[:, selected_columns]

        prediction = model.predict(preprocessed_data)

        if prediction == 0:

            st.success("""You are a low risk investor.\nLow-risk investors are typically more conservative and prioritize capital preservation over high returns. They are less willing to accept fluctuations in the value of their investments.\nHere are some investment options suitable for low-risk investors: Savings Accounts and CDs, Government Bonds, High-Quality Corporate Bonds, Money Market Funds
            """)

        elif prediction == 1:

            st.success("""You are a moderate risk investor.\nModerate-risk investors are willing to take on some level of risk in exchange for potentially higher returns than low-risk investments. They have a balanced approach to risk and return.\nSuitable investment options for moderate-risk investors include: Diversified Mutual Funds, Index Funds and ETFs, Blue-Chip Stocks, Real Estate Investment Trusts (REITs)""")

        elif prediction == 2:

            st.success("""You are a high risk investor.\nHigh-risk investors are more aggressive and seek higher returns, understanding that they may experience significant volatility and potential losses. They have a longer time horizon and are willing to take chances for potentially higher rewards.\nHere are investment options suitable for high-risk investors: Individual Stocks, Sector Funds, Commodities and Precious Metals,Venture Capital and Angel Investments
    """)
            
            
if __name__ == '__main__':
    
    main()





                          
                          
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
