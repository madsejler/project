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
import time


data = pd.read_csv("https://raw.githubusercontent.com/Ceges98/BDS-Project/main/bank_marketing.csv", sep=";")
data = data[data["education"].str.contains("unknown") == False]
data = data[data["marital"].str.contains("unknown") == False]

st.set_page_config(
    page_title = 'Data Dashboard',
    page_icon = '✅',
    layout = 'wide'
)

# dashboard title

st.title("Data Dashboard")

# top-level filters 

job_filter = st.selectbox("Select the Job", pd.unique(data['job']))


# creating a single-element container.
placeholder = st.empty()
# dataframe filter 
data = data[data['job']==job_filter]

# near real-time / live feed simulation 
for seconds in range(1):

    # creating KPIs 
    avg_age = np.mean(data['age']) 

    count_married = int(data[(data["marital"]=='married')]['marital'].count())
    
    with placeholder.container(): 
# create two columns
        age, married = st.columns(2)

        # fill in those three columns with respective metrics or KPIs 
        age.metric(label="Average Age ⏳", value=round(avg_age))
        married.metric(label="Married Count 💍", value= int(count_married))

        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("Age/Marital heatmap")
            fig = px.density_heatmap(data_frame=data, y = 'age', x = 'marital')
            st.write(fig)
        with fig_col2:
            st.markdown("Age distribution")
            fig2 = px.histogram(data_frame = data, x = 'age')
            st.write(fig2)
        st.markdown("### Detailed Data View")
        st.dataframe(data)
        time.sleep(1)
# NYT AFSNIT
st.markdown("### Nyt afsnit")
# creating a single-element container.
placeholder = st.empty()

# near real-time / live feed simulation 
for seconds in range(1):

    # creating KPIs 
    avg_rate = np.mean(data['euribor3m']) 
    
    with placeholder.container(): 
# create two columns
        eur = st.columns(1)

        # fill in those three columns with respective metrics or KPIs 
        eur.metric(label="Average Age ⏳", value=(10))

        fig_col1 = st.columns(1)
        with fig_col1:
            st.markdown("Age/Marital heatmap")
            fig = px.density_heatmap(data_frame=data, y = 'age', x = 'marital')
            st.write(fig)
        st.markdown("### Detailed Data View")
        st.dataframe(data)
        time.sleep(1)