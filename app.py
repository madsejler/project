# Load all relevant packages

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



df = pd.read_csv("https://raw.githubusercontent.com/Ceges98/BDS-Project/main/bank_marketing.csv", sep=";")
df = df[df["education"].str.contains("unknown") == False]

st.set_page_config(
    page_title = 'Real-Time Data Science Dashboard',
    page_icon = '✅',
    layout = 'wide'
)

# dashboard title

st.title("Real-Time / Live Data Science Dashboard")

# top-level filters 

job_filter = st.selectbox("Select the Job", pd.unique(df['job']))


# creating a single-element container.
placeholder = st.empty()

# dataframe filter 

df = df[df['job']==job_filter]

# near real-time / live feed simulation 

for seconds in range(200):
#while True: 
    
    df['age_new'] = df['age'] * np.random.choice(range(1,5))
    df['education_new'] = df['education'] * np.random.choice(range(1,5))

    # creating KPIs 
    avg_age = np.mean(df['age_new']) 

    count_married = int(df[(df["marital"]=='married')]['marital'].count() + np.random.choice(range(1,30)))
    

    with placeholder.container():
        # create three columns
        kpi1, kpi2 = st.columns(2)

        # fill in those three columns with respective metrics or KPIs 
        kpi1.metric(label="Age ⏳", value=round(avg_age), delta= round(avg_age) - 10)
        kpi2.metric(label="Married Count 💍", value= int(count_married), delta= - 10 + count_married)
    

        # create two columns for charts 

        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("### First Chart")
            fig = px.density_heatmap(data_frame=df, y = 'age_new', x = 'marital')
            st.write(fig)
        with fig_col2:
            st.markdown("### Second Chart")
            fig2 = px.histogram(data_frame = df, x = 'age_new')
            st.write(fig2)
        st.markdown("### Detailed Data View")
        st.dataframe(df)
        time.sleep(1)
    #placeholder.empty()
