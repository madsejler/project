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



data = pd.read_csv("https://raw.githubusercontent.com/Ceges98/BDS-Project/main/bank_marketing.csv", sep=";")
data = data[data["education"].str.contains("unknown") == False]

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

for seconds in range(200):
#while True: 
    
    data['age_new'] = data['age'] * np.random.choice(range(1,5))
    data['balance_new'] = data['balance'] * np.random.choice(range(1,5))

    # creating KPIs 
    avg_age = np.mean(data['age_new']) 

    count_married = int(data[(data["marital"]=='married')]['marital'].count() + np.random.choice(range(1,30)))
    
    balance = np.mean(data['balance_new'])
    
    with placeholder.container(): 
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
    
