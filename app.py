# Load packages (comments for more special stuff)

import pandas as pd
import pickle # un-pickling stuff from training notebook
from xgboost import XGBRegressor # we use a trained XGBoost model...and therefore need to load it
from sklearn.preprocessing import StandardScaler
import shap # add prediction explainability
import plotly.express as px

import numpy as np
import itertools # we need that to flatten ohe.categories_ into one list for columns
import streamlit as st
from streamlit_shap import st_shap # wrapper to display nice shap viz in the app

    ### EDA TAB
df = pd.read_csv("https://raw.githubusercontent.com/Ceges98/BDS-Project/main/bank_marketing.csv", sep=";")
   


st.title('Deposit Prediction for Bank Marketing Campaign')

st.write("This app is based on 16 inputs that predict wheather a customer will deposit or not? Using this app, a bank can identify specific customer segments; that will make deposits.")
st.write("Please use the following form to get started!")
st.markdown('<p class="big-font">(NOTE: For convinience, usual values are pre-selected in the form.)</p>', unsafe_allow_html=True)


    # selecting age
st.subheader("Select Customer's Age")
selected_age = st.slider("Select Customer's Age", min_value = 18, max_value = 95, 
                        step = 1, value = 41)    # Slider does not tolerate dtype value mismatch df.age.max() was thus not used.
st.write("Selected Age:", selected_age)


    # selecting job
st.subheader("Select Customer's Job")
selected_job = st.radio("", df['job'].unique(), index = 3)
st.write("Selected Job:", selected_job)


    ## Encode the job entered by user
    ### Declaring function for encoding
def encode_job(selected_item):
    dict_job = {'admin.':0, 'blue-collar':1, 'entrepreneur':2, 'housemaid':3, 'management':4, 'retired':5, 'self-employed':6, 'services':7, 'student':8, 'technician':9, 'unemployed':10, 'unknown':11}
    return dict_job.get(selected_item, 'No info available')

    ### Using function for encoding
selected_job = encode_job(selected_job)  


    # selecting marital status
st.subheader("Select Customer's Marital")
selected_marital = st.radio("", df['marital'].unique())
st.write("Selected Marital:", selected_marital)


    ## Encode the marital entered by user
    ### Declaring function for encoding
def encode_marital(selected_item):
        dict_marital = {'divorced':0, 'married':1, 'single':2}
        return dict_marital.get(selected_item, 'No info available')

    ### Using function for encoding
selected_marital = encode_marital(selected_marital)  



    # selecting education
st.subheader("Select Customer's Education")
selected_education = st.radio("", df['education'].unique())
st.write("Selected Education:", selected_education)

    ## Encode the education entered by user
    ### Declaring function for encoding
def encode_education(selected_item):
        dict_education = {'primary':0, 'secondary':1, 'tertiary':2, 'unknown':3}
        return dict_education.get(selected_item, 'No info available')

    ### Using function for encoding
selected_education = encode_education(selected_education)  


    # selecting default status
st.subheader("Select Customer's Default Status")
selected_default = st.radio("", df['default'].unique()[::-1])
st.write("Selected Default Status", selected_default)


    ## Encode the default entered by user
    ### Declaring function for encoding
def encode_default(selected_item):
        dict_default = {'no':0, 'yes':1}
        return dict_default.get(selected_item, 'No info available')

    ### Using function for encoding
selected_default = encode_default(selected_default)  