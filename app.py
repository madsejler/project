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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px
import seaborn as sns

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
                            'emp.var.rate': 1.1, #This could be scraped from a site like Statistics Portugal
                            'cons.price.idx': 93.994, #This could be scraped from a site like Statistics Portugal
                            'cons.conf.idx': -36.4, #This could be scraped from a site like Statistics Portugal
                            'euribor3m': 4.857, #This could be scraped from a site like Statistics Portugal
                            'nr.employed': 5191.0 #This could be scraped from a site like Statistics Portugal
                        }, index=[0])
    new_values_num = pd.DataFrame(scaler.transform(new_df_num), columns = new_df_num.columns, index=[0])  
    
    #bring all columns together
    line_to_pred = pd.concat([new_values_num, new_values_cat], axis=1)

    #run prediction for 1 new observation
    predicted_value = model_xgb.predict(line_to_pred)[0]
    
    
    #print out result to user 
    st.metric(label="Predicted answer", value=f'{predicted_value}')
    st.subheader(f'Why {predicted_value}? 1 equals to yes, while 0 equals to no')

    #print SHAP explainer to user
    st.subheader(f'Lets explain why the model predicts the output above! See below for SHAP value:')
    shap_value = explainer.shap_values(line_to_pred)
    st_shap(shap.force_plot(explainer.expected_value, shap_value, line_to_pred), height=400, width=900)


    st.sidebar.title('What do tou want to see?')
st.sidebar.write('Here are some options so you can navigate through this dashboard.')
option = st.sidebar.selectbox('Options:', ('Data Summary', 'Age', 'Job', 'Duration',
                                           'Education', 'Default', 'Housing',
                                           'Loan', 'Contact', 'Previous outcome',
                                           'Previous campaign', 'Campaign',
                                           'Last contact date', 'Campaign response',
                                           'Correlation heatmap'))

if option == 'Data Summary':
    st.header('Data Summary')
    st.write('Here is some summary calculations of the data, I splitted into 3 dataframes for better understanding.')
    if st.sidebar.checkbox('Raw data sets'):
        st.subheader('Bank data')
        st.dataframe(data.head())
        st.subheader('yes_df:')
        st.dataframe(yes_df.head())
        st.subheader('no_df:')
        st.dataframe(no_df.head())

    if st.sidebar.checkbox('Summary calculations'):
        st.subheader('Bank data summary:')
        st.table(bank_data.describe())
        st.subheader('yes_df summary:')
        st.table(yes_df.describe())
        st.subheader('no_df summary:')
        st.table(no_df.describe())
    else: st.write('Pick something to see in the sidebar.')

if option == 'Age':
    st.header('Age frequencies')
    st.write('Here are the age frequencies count for 3 dataframes.')
    st.subheader('Bank data')
    fig1 = plt.figure()
    age_freq = sns.countplot(x=data['age'])
    age_freq.set_xticklabels(age_freq.get_xticklabels(), rotation=40, ha="right")
    age_freq.xaxis.set_major_locator(ticker.MultipleLocator(5))
    st.pyplot(fig1)
    st.subheader('yes_df')
    fig2 = plt.figure()
    yes_age_freq = sns.countplot(x=yes_df['age'])
    yes_age_freq.set_xticklabels(yes_age_freq.get_xticklabels(), rotation=40, ha="right")
    yes_age_freq.xaxis.set_major_locator(ticker.MultipleLocator(5))
    st.pyplot(fig2)
    st.subheader('no_df')
    fig3 = plt.figure()
    no_age_freq = sns.countplot(x=no_df['age'])
    no_age_freq.set_xticklabels(no_age_freq.get_xticklabels(), rotation=40, ha="right")
    no_age_freq.xaxis.set_major_locator(ticker.MultipleLocator(5))
    st.pyplot(fig3)

if option == 'Duration':
    st.header('Duration frequencies')
    st.write('Here are the duration frequencies count for 3 dataframes.')
    st.subheader('Bank data')
    fig1 = plt.figure()
    duration_freq = sns.countplot(x=bank_data['duration'])
    duration_freq.set_xticklabels(duration_freq.get_xticklabels(), rotation=40, ha="right")
    duration_freq.xaxis.set_major_locator(ticker.MultipleLocator(100))
    st.pyplot(fig1)
    st.subheader('yes_df')
    fig2 = plt.figure()
    yes_duration_freq = sns.countplot(x=yes_df['duration'])
    yes_duration_freq.set_xticklabels(yes_duration_freq.get_xticklabels(), rotation=40, ha="right")
    yes_duration_freq.xaxis.set_major_locator(ticker.MultipleLocator(100))
    st.pyplot(fig2)
    st.subheader('no_df')
    fig3 = plt.figure()
    no_duration_freq = sns.countplot(x=no_df['duration'])
    no_duration_freq.set_xticklabels(no_duration_freq.get_xticklabels(), rotation=40, ha="right")
    no_duration_freq.xaxis.set_major_locator(ticker.MultipleLocator(100))
    st.pyplot(fig3)

if option == 'Job':
    st.header('Job frequencies')
    st.write('Here are the job frequencies count for 3 dataframes.')
    st.subheader('Bank data')
    fig1 = plt.figure()
    job_freq = sns.countplot(x=bank_data['job'])
    job_freq.set_xticklabels(job_freq.get_xticklabels(), rotation=40, ha="right")
    st.pyplot(fig1)
    st.subheader('yes_df')
    fig2 = plt.figure()
    yes_job_freq = sns.countplot(x=yes_df['job'])
    yes_job_freq.set_xticklabels(yes_job_freq.get_xticklabels(), rotation=40, ha="right")
    st.pyplot(fig2)
    st.subheader('no_df')
    fig3 = plt.figure()
    no_job_freq = sns.countplot(x=no_df['job'])
    no_job_freq.set_xticklabels(no_job_freq.get_xticklabels(), rotation=40, ha="right")
    st.pyplot(fig3)

if option == 'Education':
    st.header('Education frequencies')
    st.write('Here are the education frequencies count for 3 dataframes.')
    st.subheader('Bank data')
    fig1 = plt.figure()
    education_freq = sns.countplot(x=bank_data['education'])
    education_freq.set_xticklabels(education_freq.get_xticklabels(), rotation=40, ha="right")
    st.pyplot(fig1)
    st.subheader('yes_df')
    fig2 = plt.figure()
    yes_education_freq = sns.countplot(x=yes_df['education'])
    yes_education_freq.set_xticklabels(yes_education_freq.get_xticklabels(), rotation=40, ha="right")
    st.pyplot(fig2)
    st.subheader('no_df')
    fig3 = plt.figure()
    no_education_freq = sns.countplot(x=no_df['education'])
    no_education_freq.set_xticklabels(no_education_freq.get_xticklabels(), rotation=40, ha="right")
    st.pyplot(fig3)

if option == 'Default':
    st.header('Default frequencies')
    st.write('Here are the default frequencies count for 3 dataframes.')
    st.subheader('Bank data')
    fig1 = plt.figure()
    default_freq = sns.countplot(x=bank_data['default'])
    st.pyplot(fig1)
    st.subheader('yes_df')
    fig2 = plt.figure()
    yes_default_freq = sns.countplot(x=yes_df['default'])
    st.pyplot(fig2)
    st.subheader('no_df')
    fig3 = plt.figure()
    no_default_freq = sns.countplot(x=no_df['default'])
    st.pyplot(fig3)

if option == 'Housing':
    st.header('Housing frequencies')
    st.write('Here are the housing frequencies count for 3 dataframes.')
    st.subheader('Bank data')
    fig1 = plt.figure()
    housing_freq = sns.countplot(x=bank_data['housing'])
    st.pyplot(fig1)
    st.subheader('yes_df')
    fig2 = plt.figure()
    yes_housing_freq = sns.countplot(x=yes_df['housing'])
    st.pyplot(fig2)
    st.subheader('no_df')
    fig3 = plt.figure()
    no_housing_freq = sns.countplot(x=no_df['housing'])
    st.pyplot(fig3)

if option == 'Loan':
    st.header('Loan frequencies')
    st.write('Here are the loan frequencies count for 3 dataframes.')
    st.subheader('Bank data')
    fig1 = plt.figure()
    loan_freq = sns.countplot(x=bank_data['loan'])
    st.pyplot(fig1)
    st.subheader('yes_df')
    fig2 = plt.figure()
    yes_loan_freq = sns.countplot(x=yes_df['loan'])
    st.pyplot(fig2)
    st.subheader('no_df')
    fig3 = plt.figure()
    no_loan_freq = sns.countplot(x=no_df['loan'])
    st.pyplot(fig3)

if option == 'Contact':
    st.header('Contact frequencies')
    st.write('Here are the contact frequencies count for 3 dataframes.')
    st.subheader('Bank data')
    fig1 = plt.figure()
    contact_freq = sns.countplot(x=bank_data['contact'])
    st.pyplot(fig1)
    st.subheader('yes_df')
    fig2 = plt.figure()
    yes_contact_freq = sns.countplot(x=yes_df['contact'])
    st.pyplot(fig2)
    st.subheader('no_df')
    fig3 = plt.figure()
    no_contact_freq = sns.countplot(x=no_df['contact'])
    st.pyplot(fig3)

if option == 'Previous outcome':
    st.header('Previous outcome frequencies')
    st.write('Here are the previous outcome frequencies count for 3 dataframes.')
    st.subheader('Bank data')
    fig1 = plt.figure()
    poutcome_freq = sns.countplot(x=bank_data['poutcome'])
    st.pyplot(fig1)
    st.subheader('yes_df')
    fig2 = plt.figure()
    yes_poutcome_freq = sns.countplot(x=yes_df['poutcome'])
    st.pyplot(fig2)
    st.subheader('no_df')
    fig3 = plt.figure()
    no_poutcome_freq = sns.countplot(x=no_df['poutcome'])
    st.pyplot(fig3)

if option == 'Previous campaign':
    st.header('Previous campaign frequencies')
    st.write('Here are the previous campaign frequencies count for 3 dataframes.')
    st.subheader('Bank data')
    fig1 = plt.figure()
    previous_freq = sns.countplot(x=bank_data['previous'])
    previous_freq.set_xticklabels(previous_freq.get_xticklabels(), rotation=40, ha="right")
    previous_freq.xaxis.set_major_locator(ticker.MultipleLocator(3))
    st.pyplot(fig1)
    st.subheader('yes_df')
    fig2 = plt.figure()
    yes_previous_freq = sns.countplot(x=yes_df['previous'])
    yes_previous_freq.set_xticklabels(yes_previous_freq.get_xticklabels(), rotation=40, ha="right")
    yes_previous_freq.xaxis.set_major_locator(ticker.MultipleLocator(3))
    st.pyplot(fig2)
    st.subheader('no_df')
    fig3 = plt.figure()
    no_previous_freq = sns.countplot(x=no_df['previous'])
    no_previous_freq.set_xticklabels(no_previous_freq.get_xticklabels(), rotation=40, ha="right")
    no_previous_freq.xaxis.set_major_locator(ticker.MultipleLocator(3))
    st.pyplot(fig3)

if option == 'Campaign':
    st.header('Campaign frequencies')
    st.write('Here are the campaign frequencies count for 3 dataframes.')
    st.subheader('Bank data')
    fig1 = plt.figure()
    campaign_freq = sns.countplot(x=bank_data['campaign'])
    campaign_freq.set_xticklabels(campaign_freq.get_xticklabels(), rotation=40, ha="right")
    campaign_freq.xaxis.set_major_locator(ticker.MultipleLocator(3))
    st.pyplot(fig1)
    st.subheader('yes_df')
    fig2 = plt.figure()
    yes_campaign_freq = sns.countplot(x=yes_df['campaign'])
    yes_campaign_freq.set_xticklabels(yes_campaign_freq.get_xticklabels(), rotation=40, ha="right")
    yes_campaign_freq.xaxis.set_major_locator(ticker.MultipleLocator(3))
    st.pyplot(fig2)
    st.subheader('no_df')
    fig3 = plt.figure()
    no_campaign_freq = sns.countplot(x=no_df['campaign'])
    no_campaign_freq.set_xticklabels(no_campaign_freq.get_xticklabels(), rotation=40, ha="right")
    no_campaign_freq.xaxis.set_major_locator(ticker.MultipleLocator(3))
    st.pyplot(fig3)

if option == 'Last contact date':
    st.header('Last contact date frequencies')
    st.write('Here are the last contact frequencies count for 3 dataframes.')
    st.subheader('Bank data')
    df_count = bank_data[['lc_date', 'y']].groupby(['lc_date']).count().reset_index()
    df_count_date = px.histogram(x=df_count['lc_date'],
                                 y=df_count['y'])
    st.plotly_chart(df_count_date)
    st.subheader('yes_df')
    yes_df_count = yes_df.groupby(by=["lc_date", "y"]).size().reset_index(name="counts")
    yes_df_count_date = px.histogram(x = yes_df_count['lc_date'], y = yes_df_count['counts'])
    st.plotly_chart(yes_df_count_date)
    st.subheader('no_df')
    no_df_count = no_df.groupby(by=["lc_date", "y"]).size().reset_index(name="counts")
    no_df_count_date = px.histogram(x = no_df_count['lc_date'], y = no_df_count['counts'])
    st.plotly_chart(no_df_count_date)

if option == 'Campaign response':
    st.subheader('Campaign response')
    st.write('Here is the response from the customers to the campaign.')
    fig1 = plt.figure()
    y_freq = sns.countplot(x = bank_data['y'])
    st.pyplot(fig1)

if option == 'Correlation heatmap':
    st.subheader('Correlation heatmap')
    st.write('Here is the correlation matrix plotted in a heatmap for easier visualization.')
    fig1 = plt.figure()
    corr_map = sns.heatmap(bank_data.corr(), annot=True)
    st.pyplot(fig1)