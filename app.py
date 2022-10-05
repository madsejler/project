# Load packages (comments for more special stuff)

import pandas as pd
import pickle # un-pickling stuff from training notebook
from xgboost import XGBRegressor # we use a trained XGBoost model...and therefore need to load it
from sklearn.preprocessing import StandardScaler
import shap # add prediction explainability

import numpy as np
import itertools # we need that to flatten ohe.categories_ into one list for columns
import streamlit as st
from streamlit_shap import st_shap # wrapper to display nice shap viz in the app

from datetime import datetime

st.set_page_config(
    page_title="Bank marketing prediction")

st.title('Will this given costumer say yes?')

#this is how you can add images e.g. from unsplash (or loca image file)
#st.image('https://source.unsplash.com/0PSCd1wIrm4', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

data = pd.read_csv("https://raw.githubusercontent.com/Ceges98/BDS-Project/main/bank_marketing.csv", sep=";")
data = data[data["education"].str.contains("unknown") == False]

# use this decorator (--> @st.experimental_singleton) and 0-parameters function to only load and preprocess once
@st.experimental_singleton
def read_objects():
    model_xgb = pickle.load(open('model_xgb.pkl','rb'))
    scaler = pickle.load(open('scaler.pkl','rb'))
    ohe = pickle.load(open('ohe.pkl','rb'))
    shap_values = pickle.load(open('shap_values.pkl','rb'))
    cats = list(itertools.chain(*ohe.categories_))
    return model_xgb, scaler, ohe, cats, shap_values

model_xgb, scaler, ohe, cats, shap_values = read_objects()

# define explainer
explainer = shap.TreeExplainer(model_xgb)

#write some markdown blah
with st.expander("What's that app?"):
    st.markdown("""
    This app will help you determine what you should be asking people to pay per night for staying at your awesome place.
    We trained an AI on successful places in Copenhagen. It will give you a pricing suggestion given a few inputs.
    We recommend going around 350kr up or down depending on the amenities that you can provide and the quality of your place.
    As a little extra 🌟, we added an AI explainer 🤖 to understand factors driving prices up or down.
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

# make a nice button that triggers creation of a new data-line in the format that the model expects and prediction
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
                            'emp.var.rate': 1.1, #This could be withdrawn from a site like Statistics Portugal
                            'cons.price.idx': 93.994, #This could be withdrawn from a site like Statistics Portugal
                            'cons.conf.idx': -36.4, #This could be withdrawn from a site like Statistics Portugal
                            'euribor3m': 4.857, #This could be withdrawn from a site like Statistics Portugal
                            'nr.employed': 5191.0 #This could be withdrawn from a site like Statistics Portugal
                        }, index=[0])
    new_values_num = pd.DataFrame(scaler.transform(new_df_num), columns = new_df_num.columns, index=[0])  
    
    #bring all columns together
    line_to_pred = pd.concat([new_values_num, new_values_cat], axis=1)

    #run prediction for 1 new observation
    predicted_value = model_xgb.predict(line_to_pred)[0]

    #print out result to user
    st.metric(label="Predicted answer", value=f'{predicted_value}')
    #if st.metric:
        #if(predicted_value == 1):
        #st.success('This customer segment will Deposit')
        #else:
        #st.success('This customer segment will NOT Deposit')  
    
    #print SHAP explainer to user
    st.subheader(f'Why {predicted_value}? See below:')
    shap_value = explainer.shap_values(line_to_pred)
    st_shap(shap.force_plot(explainer.expected_value, shap_value, line_to_pred), height=400, width=500)