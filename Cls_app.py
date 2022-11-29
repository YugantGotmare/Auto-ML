import streamlit as st
from streamlit_option_menu import option_menu
from pycaret.classification import setup, compare_models, pull, save_model, load_model 
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    
    selected = option_menu('AutoML',
                          
                          ['Upload Your Data',
                           'EDA',
                           'ML Model',
                           'Download Model'],
                          default_index=0)



if selected == "Upload Your Data":
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset")
        if file: 
            df = pd.read_csv(file, index_col=None)
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)

if selected == "EDA": 
    st.title("Exploratory Data Analysis (EDA)")
    profile_df = df.profile_report()
    st_profile_report(profile_df)            


                
if selected == "ML Model": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'Your_Model')


if selected == "Download Model": 
    st.title("Download Your Model")
    with open('Your_Model.pkl', 'rb') as f: 
        st.download_button('Download', f, file_name="Your_model.pkl")

   