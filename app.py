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

st.set_page_config(
    page_title="Bank marketing prediction")

st.title('Will this given costumer say yes?')

#this is how you can add images e.g. from unsplash (or loca image file)
#st.image('https://source.unsplash.com/0PSCd1wIrm4', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

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

st.subheader('Costumer description')

# here you collect all inputs from the user into new objects
job = st.selectbox('What is his/hers jobtype?', options=ohe.categories_[0])
marital = st.radio('Marital', options=ohe.categories_[1])
poutcome = st.selectbox('What was the previous outcome for this costumer?', options=ohe.categories_[4])
age = st.number_input('Age?', min_value=17, max_value=98)
education = st.number_input('Education', min_value=0, max_value=7)
campaign = st.number_input('How many contacts have you made for this costumer for this campagin already?', min_value=0, max_value=35)
previous = st.number_input('How many times have you contacted this client before?', min_value=0, max_value=35)

# make a nice button that triggers creation of a new data-line in the format that the model expects and prediction
if st.button('Predict! 🚀'):
    # make a DF for categories and transform with one-hot-encoder
    new_df_cat = pd.DataFrame({'job':job,
                'marital':marital,
                'month': 'oct',
                'day_of_week':'fri',
                'poutcome':poutcome}, index=[0])
    new_values_cat = pd.DataFrame(ohe.transform(new_df_cat), columns = cats , index=[0])

    # make a DF for the numericals and standard scale
    new_df_num = pd.DataFrame({'age':age, 
                            'education': education,
                            'campaign': campaign,
                            'previous': previous, 
                            'emp.var.rate': 1.1,
                            'cons.price.idx': 93.994,
                            'cons.conf.idx': -36.4,
                            'euribor3m': 4.857,
                            'nr.employed': 5191.0
                        }, index=[0])
    new_values_num = pd.DataFrame(scaler.transform(new_df_num), columns = new_df_num.columns, index=[0])  
    
    #bring all columns together
    line_to_pred = pd.concat([new_values_num, new_values_cat], axis=1)

    #run prediction for 1 new observation
    predicted_value = model_xgb.predict(line_to_pred)[0]

    #print out result to user
    st.metric(label="Predicted answer", value=f'{predicted_value}')
    
    #print SHAP explainer to user
    st.subheader(f'Why {predicted_value}? See below:')
    shap_value = explainer.shap_values(line_to_pred)
    st_shap(shap.force_plot(explainer.expected_value, shap_value, line_to_pred), height=400, width=500)


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