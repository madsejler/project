# Load packages (comments for more special stuff)

import pandas as pd
import pickle # un-pickling stuff from training notebook
from xgboost import XGBRegressor # we use a trained XGBoost model...and therefore need to load it
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import shap # add prediction explainability

import numpy as np
import itertools # we need that to flatten ohe.categories_ into one list for columns
import streamlit as st
from streamlit_shap import st_shap # wrapper to display nice shap viz in the app
import plotly.express as px
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

##Streamlit interface:
st.set_page_config(page_title='Bank Marketing Project',
                    page_icon="🐙",
                    layout='wide')

colT1,colT2 = st.columns([10,20])
with colT2:
   st.title('Bank Markerting Project 💣💥')

data = pd.read_csv("https://raw.githubusercontent.com/Ceges98/BDS-Project/main/bank_marketing.csv", sep=";")
data = data[data["education"].str.contains("unknown") == False]
data = data[data["marital"].str.contains("unknown") == False]
data = data[data["job"].str.contains("unknown") == False]
        st.title('Will this given costumer say yes?')

        #st.image('https://source.unsplash.com/WgUHuGSWPVM', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")



        @st.experimental_singleton
        def read_objects():
            model_xgb = pickle.load(open('model_xgb.pkl','rb'))
            scaler = pickle.load(open('scaler.pkl','rb'))
            ohe = pickle.load(open('ohe.pkl','rb'))
            shap_values = pickle.load(open('shap_values.pkl','rb'))
            cats = list(itertools.chain(*ohe.categories_))
            return model_xgb, scaler, ohe, cats, shap_values

        model_xgb, scaler, ohe, cats, shap_values = read_objects()

        #Explainer defined
        explainer = shap.TreeExplainer(model_xgb)

        with st.expander("What's the purpose of this app?"):
            st.markdown("""
            This app will help you determine if you should call a given costumer! 💵 💴 💶 💷
            It can further help you reconsider your strategic approach to the costumer,
            in the case that our SML model will predict a "No" from the costumer.
            """)

        st.title('Costumer description')

        #Below all the bank client's info will be selected
        st.subheader("Select the Customer's Age")
        age = st.slider("", min_value = 17, max_value = 98, 
                                step = 1, value = 41)
        st.write("Selected Age:", age)

        st.subheader("Select the Customer's Jobtype")
        job = st.radio("", ohe.categories_[0])
        st.write("Selected Job:", job)

        st.subheader("Select the Customer's Marital")
        marital = st.radio("", ohe.categories_[1])
        st.write("Selected Marital:", marital)

        st.subheader("Select the Customer's Education")
        education = st.radio("", data['education'].unique())
        st.write("Selected Education:", education)
        #Defining a encoding function for education
        def encode_education(selected_item):
            dict_education = {'basic.4y':1, 'high.school':4, 'basic.6y':2, 'basic.9y':3, 'professional.course':5, 'university.degree':6, 
        'illiterate':0}
            return dict_education.get(selected_item)
        ### Using function for encoding on education
        education = encode_education(education) 

        poutcome = st.selectbox('What was the previous outcome for this costumer?', options=ohe.categories_[4])
        campaign = st.number_input('How many contacts have you made for this costumer for this campagin already?', min_value=0, max_value=35)
        previous = st.number_input('How many times have you contacted this client before?', min_value=0, max_value=35)

        #Button for predicting the costumers answer
        if st.button('Deposit Prediction 💵'):

            # make a DF for categories and transform with one-hot-encoder
            new_df_cat = pd.DataFrame({'job':job,
                        'marital':marital,
                        'month': 'oct', #This could be coded with a date.today().month function
                        'day_of_week':'fri', #This could aswell be coded with a function
                        'poutcome':poutcome}, index=[0])
            new_values_cat = pd.DataFrame(ohe.transform(new_df_cat), columns = cats , index=[0])

            # make a DF for the numericals and standard scale
            new_df_num = pd.DataFrame({'age':age, 
                                    'education': education,
                                    'campaign': campaign,
                                    'previous': previous, 
                                    'emp.var.rate': 1.1, #This could be scraped from a site like Statistics Portugal
                                    'cons.price.idx': 93.994, #This could be scraped from a site like Statistics Portugal
                                    'cons.conf.idx': -36.4, #This could be scraped from a site like Statistics Portugal
                                    'euribor3m': 4.857, #This could be scraped from a site like Statistics Portugal
                                    'nr.employed': 5191.0 #This could be scraped from a site like Statistics Portugal
                                }, index=[0])
            new_values_num = pd.DataFrame(scaler.transform(new_df_num), columns = new_df_num.columns, index=[0])  
            
            #Bringing all columns together
            line_to_pred = pd.concat([new_values_num, new_values_cat], axis=1)

            #Run prediction for the new observation. Inputs to this given above
            predicted_value = model_xgb.predict(line_to_pred)[0]
            
            
            #Printing the result
            st.metric(label="Predicted answer", value=f'{predicted_value}')
            st.subheader(f'What does {predicted_value} mean? 1 equals to yes, while 0 equals to no')

            #Printing SHAP explainer
            st.subheader(f'Lets explain why the model predicts the output above! See below for SHAP value:')
            shap_value = explainer.shap_values(line_to_pred)
            st_shap(shap.force_plot(explainer.expected_value, shap_value, line_to_pred), height=400, width=900)