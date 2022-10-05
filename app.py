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
st.set_page_config(layout = "wide", page_icon = 'logo.png', page_title='EDA')

st.header("🎨Exploratory Data Analysis Tool for Data Science Projects")

st.write('<p style="font-size:160%">You will be able to✅:</p>', unsafe_allow_html=True)

st.write('<p style="font-size:100%">&nbsp 1. See the whole dataset</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%">&nbsp 2. Get column names, data types info</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%">&nbsp 3. Get the count and percentage of NA values</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%">&nbsp 4. Get descriptive analysis </p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%">&nbsp 5. Check inbalance or distribution of target variable:</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%">&nbsp 6. See distribution of numerical columns</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%">&nbsp 7. See count plot of categorical columns</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%">&nbsp 8. Get outlier analysis with box plots</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%">&nbsp 9. Obtain info of target value variance with categorical columns</p>', unsafe_allow_html=True)
#st.image('header2.png', use_column_width = True)

functions.space()
st.write('<p style="font-size:130%">Import Dataset</p>', unsafe_allow_html=True)

file_format = st.radio('Select file format:', ('csv', 'excel'), key='file_format')
dataset = st.file_uploader(label = '')

use_defo = st.checkbox('Use example Dataset')
if use_defo:
    dataset = 'CarPrice_Assignment.csv'

st.sidebar.header('Import Dataset to Use Available Features: 👉')

if dataset:
    if file_format == 'csv' or use_defo:
        df = pd.read_csv(dataset)
    else:
        df = pd.read_excel(dataset)
    st.subheader('Dataframe:')
    n, m = df.shape
    st.write(f'<p style="font-size:130%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)   
    st.dataframe(df)


    all_vizuals = ['Info', 'NA Info', 'Descriptive Analysis', 'Target Analysis', 
                   'Distribution of Numerical Columns', 'Count Plots of Categorical Columns', 
                   'Box Plots', 'Outlier Analysis', 'Variance of Target with Categorical Columns']
    functions.sidebar_space(3)         
    vizuals = st.sidebar.multiselect("Choose which visualizations you want to see 👇", all_vizuals)

    if 'Info' in vizuals:
        st.subheader('Info:')
        c1, c2, c3 = st.columns([1, 2, 1])
        c2.dataframe(functions.df_info(df))

    if 'NA Info' in vizuals:
        st.subheader('NA Value Information:')
        if df.isnull().sum().sum() == 0:
            st.write('There is not any NA value in your dataset.')
        else:
            c1, c2, c3 = st.columns([0.5, 2, 0.5])
            c2.dataframe(functions.df_isnull(df), width=1500)
            functions.space(2)
            

    if 'Descriptive Analysis' in vizuals:
        st.subheader('Descriptive Analysis:')
        st.dataframe(df.describe())
        
    if 'Target Analysis' in vizuals:
        st.subheader("Select target column:")    
        target_column = st.selectbox("", df.columns, index = len(df.columns) - 1)
    
        st.subheader("Histogram of target column")
        fig = px.histogram(df, x = target_column)
        c1, c2, c3 = st.columns([0.5, 2, 0.5])
        c2.plotly_chart(fig)


    num_columns = df.select_dtypes(exclude = 'object').columns
    cat_columns = df.select_dtypes(include = 'object').columns

    if 'Distribution of Numerical Columns' in vizuals:

        if len(num_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = functions.sidebar_multiselect_container('Choose columns for Distribution plots:', num_columns, 'Distribution')
            st.subheader('Distribution of numerical columns')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_num_cols)):
                        break

                    fig = px.histogram(df, x = selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width = True)
                    i += 1

    if 'Count Plots of Categorical Columns' in vizuals:

        if len(cat_columns) == 0:
            st.write('There is no categorical columns in the data.')
        else:
            selected_cat_cols = functions.sidebar_multiselect_container('Choose columns for Count plots:', cat_columns, 'Count')
            st.subheader('Count plots of categorical columns')
            i = 0
            while (i < len(selected_cat_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_cat_cols)):
                        break

                    fig = px.histogram(df, x = selected_cat_cols[i], color_discrete_sequence=['indianred'])
                    j.plotly_chart(fig)
                    i += 1

    if 'Box Plots' in vizuals:
        if len(num_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = functions.sidebar_multiselect_container('Choose columns for Box plots:', num_columns, 'Box')
            st.subheader('Box plots')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:
                    
                    if (i >= len(selected_num_cols)):
                        break
                    
                    fig = px.box(df, y = selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width = True)
                    i += 1

    if 'Outlier Analysis' in vizuals:
        st.subheader('Outlier Analysis')
        c1, c2, c3 = st.columns([1, 2, 1])
        c2.dataframe(functions.number_of_outliers(df))

    if 'Variance of Target with Categorical Columns' in vizuals:
        
        
        df_1 = df.dropna()
        
        high_cardi_columns = []
        normal_cardi_columns = []

        for i in cat_columns:
            if (df[i].nunique() > df.shape[0] / 10):
                high_cardi_columns.append(i)
            else:
                normal_cardi_columns.append(i)


        if len(normal_cardi_columns) == 0:
            st.write('There is no categorical columns with normal cardinality in the data.')
        else:
        
            st.subheader('Variance of target variable with categorical columns')
            model_type = st.radio('Select Problem Type:', ('Regression', 'Classification'), key = 'model_type')
            selected_cat_cols = functions.sidebar_multiselect_container('Choose columns for Category Colored plots:', normal_cardi_columns, 'Category')
            
            if 'Target Analysis' not in vizuals:   
                target_column = st.selectbox("Select target column:", df.columns, index = len(df.columns) - 1)
            
            i = 0
            while (i < len(selected_cat_cols)):
                
                
            
                if model_type == 'Regression':
                    fig = px.box(df_1, y = target_column, color = selected_cat_cols[i])
                else:
                    fig = px.histogram(df_1, color = selected_cat_cols[i], x = target_column)

                st.plotly_chart(fig, use_container_width = True)
                i += 1

            if high_cardi_columns:
                if len(high_cardi_columns) == 1:
                    st.subheader('The following column has high cardinality, that is why its boxplot was not plotted:')
                else:
                    st.subheader('The following columns have high cardinality, that is why its boxplot was not plotted:')
                for i in high_cardi_columns:
                    st.write(i)
                
                st.write('<p style="font-size:140%">Do you want to plot anyway?</p>', unsafe_allow_html=True)    
                answer = st.selectbox("", ('No', 'Yes'))

                if answer == 'Yes':
                    for i in high_cardi_columns:
                        fig = px.box(df_1, y = target_column, color = i)
                        st.plotly_chart(fig, use_container_width = True)