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


tab1, tab2, tab3, tab4 = st.tabs(["Data Exploration","Predictor tool SML", "SML Model Comparison", "UML"])
with tab1:

    # dashboard title

    st.title("Data Dashboard")

    # top-level filters 

    job_filter = st.selectbox("Select the Job", pd.unique(data['job']))


    # creating a single-element container.
    placeholder = st.empty()
    # dataframe filter 
    data = data[data['job']==job_filter]

    # near real-time / live feed simulation 
    for seconds in range(10):

        # creating metrices 
        avg_age = np.mean(data['age']) 

        count_married = int(data[(data["marital"]=='married')]['marital'].count())
        
        with placeholder.container(): 
    # create two columns
            age, married = st.columns(2)

            # fill in those two columns with respective metrics 
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
    with tab2:
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

        with tab3:
            st.subheader("SML Model Accuracy")
            st.markdown("On this tab, we will explain why we used the XGB-model, and what parameters we made the decision on")
            with st.expander("What is the method for comparing the choosing of the three models?"):
                st.markdown(""" The method for comparing, is done through running the notebook five times 
                and chekcing how much the accuracy of the three models each time. We end up with five values
                for each model, and then we are comparing the mean, to which model overall performs the best.
                There are several random elements in the code, for instance we undersample the data in order
                to balance it for SML purposes. The undersample is done randomly and everytime we run the code
                the undersample will include different obersvations. Further the train-test split consist of a 
                random element. By running the note 5 times we expect the deviation to be under of the accuracy
                to be under 5%. Take a look at this page to see the documentation for the values of the 5 runs.
                """)
            st.subheader("Logistic Regression")
            st.markdown("The five values for this model had a range of 0,78% which a good deal under the 5% 🤓 ")

            st.subheader("XGBClassifier")
            st.markdown("The five values for this model had a range of 2,23% which is also under 5% 🤗 ")

            st.subheader("RandomForrester")
            st.markdown("The five values for this model had a range of 1,23% which is also under 5% 🎉 ")

            st.subheader("Ranking of the models by the mean accuracy of the 5 runs")
            st.markdown("1. **Logistic Regression**: 74,50% 🏆" )
            st.markdown("2. **XGB Classifier**: 73,44% 🥈" )
            st.markdown("3. **Random Forest**: 71,30% 🥉" )

            st.markdown("""Due to some technical issues with the Logistic regression, we decided to use the XGB Classifier
            for the model anyways, because the LR-model seems to do limited ietrations on the training data. We did not 
            have that problem with the XGB-model, so we went ahead and used the XGB for the prediction model on this webpage """)

        with tab4: 
            data1 = pd.read_csv("https://raw.githubusercontent.com/Ceges98/BDS-Project/main/bank_marketing.csv", sep=";")
            with st.expander("UML"):
                st.title("Unsupervised Machine Learning")
                st.subheader('This will be a journey through the creation of UML customer segmentation, and an analysis of the obtained result.')
                'Let us start with the end result'
                st.image('https://raw.githubusercontent.com/Ceges98/BDS-Project/main/visualization.png', caption='not an optimal result')
                st.subheader('How did this come to be?')
                'To start the process of customer segmentation we need data regarding them.'
                data_raw = data.iloc[:, 0:7]
                st.write(data_raw.head(100))
                st.caption('these are the first 100 entrances in our relevant dataset, currently unfiltered.')
                'Some work is needed for this data to be operable in regards to UML, first we remove the unknown'
                data_raw = data_raw[data_raw["job"].str.contains("unknown") == False]
                data_raw = data_raw[data_raw["marital"].str.contains("unknown") == False]
                data_raw = data_raw[data_raw["education"].str.contains("unknown") == False]
                data_raw = data_raw[data_raw["housing"].str.contains("unknown") == False]
                data_raw = data_raw[data_raw["loan"].str.contains("unknown") == False]
                data_raw.drop('default', inplace=True, axis=1)
                data = data_raw
                tab01, tab02 = st.tabs(['new data', 'code'])
                with tab01:
                    st.write(data_raw.head(50))
                    st.caption('now there are no unknown values, we have also dropped the default column as it is almost solely "no" values and therefore should not be used to segment the customers.')
                with tab02:
                    drop_unknown = '''data_raw = data_raw[data_raw["job"].str.contains("unknown") == False]
                data_raw = data_raw[data_raw["marital"].str.contains("unknown") == False]
                data_raw = data_raw[data_raw["education"].str.contains("unknown") == False]
                data_raw = data_raw[data_raw["housing"].str.contains("unknown") == False]
                data_raw = data_raw[data_raw["loan"].str.contains("unknown") == False]
                data_raw.drop('default', inplace=True, axis=1)''' 
                    st.code(drop_unknown, language='python')
                'Next up is the fact that our data is unusable due to it being in a non-numerical format'
                'To fix this spread age out into 4 categories, replace yes/no with 1/0 on housing/loan, LabelEncode education and make a, admittedly subjective, list for jobs based on income'
                def age(data_raw):
                    data_raw.loc[data_raw['age'] <= 30, 'age'] = 1
                    data_raw.loc[(data_raw['age'] > 30) & (data_raw['age'] <= 45), 'age'] = 2
                    data_raw.loc[(data_raw['age'] > 45) & (data_raw['age'] <= 65), 'age'] = 3
                    data_raw.loc[(data_raw['age'] > 65) & (data_raw['age'] <= 98), 'age'] = 4 
                    return data_raw
                age(data_raw);
                data_raw = data_raw.replace(to_replace=['yes', 'no'], value=[1, 0])
                data_raw = data_raw.replace(to_replace=['unemployed', 'student', 'housemaid', 'blue-collar', 'services', 'retired', 'technician', 'admin.', 'self-employed', 'entrepreneur', 'management'], value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                labelencoder_X = LabelEncoder()  
                data_raw['education'] = data_raw['education'].replace({'illiterate':'a_illiterate'})
                data_raw['education'] = labelencoder_X.fit_transform(data_raw['education'])
                tab03, tab04 = st.tabs(['numeric data', 'code'])
                with tab03:
                    st.write(data_raw.head(50))
                
                with tab04:
                    numerification = '''def age(data_raw):
                    data_raw.loc[data_raw['age'] <= 30, 'age'] = 1
                    data_raw.loc[(data_raw['age'] > 30) & (data_raw['age'] <= 45), 'age'] = 2
                    data_raw.loc[(data_raw['age'] > 45) & (data_raw['age'] <= 65), 'age'] = 3
                    data_raw.loc[(data_raw['age'] > 65) & (data_raw['age'] <= 98), 'age'] = 4 
                    return data_raw
                age(data_raw);
                data_raw = data_raw.replace(to_replace=['yes', 'no'], value=[1, 0])
                data_raw = data_raw.replace(to_replace=['unemployed', 'student', 'housemaid', 'blue-collar', 'services', 'retired', 'technician', 'admin.', 'self-employed', 'entrepreneur', 'management'], value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                labelencoder_X = LabelEncoder()  
                data_raw['education'] = data_raw['education'].replace({'illiterate':'a_illiterate'})
                data_raw['education'] = labelencoder_X.fit_transform(data_raw['education'])'''
                    st.code(numerification, language='python')
                st.caption('this is not a perfect way of handling the issue but onehotencoding gave rise to different issues.')
                'It may be noted that marriage is currently untouched, this is due to troubles with OneHotEncoding. As such is was deemed unwise to throw in yet another subjective variable. It will therefor be dropped.'
                data_raw = data_raw.drop(columns = 'marital')
                st.write(data_raw.head())
                'lastly these numbers need to be scaled'
                data_raw_scaled = scaler.fit_transform(data_raw)
                tab05, tab06 = st.tabs(['scaled data', 'code'])
                with tab05:
                    st.write(data_raw_scaled[:10])
                with tab06:
                    scaled_date = '''data_raw_scaled = scaler.fit_transform(data_raw)'''
                    st.code(scaled_date, language='python')
                st.caption('Now the previous sizes of the values have been standard scaled.')
                'From here on out the process will be shown through code with comments'
                rest = '''#umap accepts standard-scaled data
    embeddings = umap_scaler.fit_transform(data_raw_scaled)
    #we choose 6 clusters
    clusterer = KMeans(n_clusters=6)
    Sum_of_squared_distances = []
    K = range(1,10)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data_raw_scaled)
        Sum_of_squared_distances.append(km.inertia_)
    #no clear elbow
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    #we fit clusters on our scaled data
    clusterer.fit(data_raw_scaled)
    #we then copy the clusters into the original file
    data1['cluster'] = clusterer.labels_
    #can use the clusters to fx. see the mean of age in our clusters.
    #note that age does not seem a big factor in clustering as the mean is mostly the same.
    data1.groupby('cluster').age.mean()
    #prepping our vis_data
    vis_data = pd.DataFrame(embeddings)
    vis_data['cluster'] = data1['cluster']
    vis_data['education'] = data1['education']
    vis_data['age'] = data1['age']
    vis_data['job'] = data1['job']
    vis_data['marital'] = data1['marital']
    vis_data['housing'] = data1['housing']
    vis_data['loan'] = data1['loan']
    vis_data.columns = ['x', 'y', 'cluster','education', 'age', 'job', 'marital', 'housing', 'loan']
    #finally plotting the data with relevant tooltips
    #for unknown reasons a null cluster is made alongside our other clusters
    alt.data_transformers.enable('default', max_rows=None)
    alt.Chart(vis_data).mark_circle(size=60).encode(
        x='x',
        y='y',
        tooltip=['education', 'age', 'job', 'marital', 'housing', 'loan'],
        color=alt.Color('cluster:N', scale=alt.Scale(scheme='dark2')) #use N after the var to tell altair that it's categorical
    ).interactive()'''
                st.code(rest, language='python')
                'The reasoning behind showing this block of code is mainly to show the procedure that was taken following the data-preprocessing and showing a more in-depth process is not very useful as the end result is flawed.'
                'Speaking of, here we have once again the result so that the flaws can be discussed'
                st.image('https://raw.githubusercontent.com/Ceges98/BDS-Project/main/visualization.png', caption='still not optimal')
                '''To understand the flaws we have to look at the goal of the model. 
                The goal of this model was to place the customers in clusters based on their data.
                As such there are two problems:
                1. The clusters are randomly dispersed.
                2. An extra null-cluster has been created.
                Optimally we would be able to find and fix the problem causing these flaws but as of know this model
                has presented a learning opportunity and not a finished piece of work.'''