# import all the packages
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# build the user interface
st.set_page_config(page_title="Iris Project", layout='wide')

# add title
st.title('Iris Project - Koustubh Jagtap')

# Add input for the user
sep_len = st.number_input('Sepal Length : ', min_value=0.00, step=0.01)
sep_wid = st.number_input('Sepal Width : ', min_value=0.00, step=0.01)
pet_len = st.number_input('Petal Length : ', min_value=0.00, step=0.01)
pet_wid = st.number_input('Petal Width : ', min_value=0.00, step=0.01)

# Add a button to predict
submit = st.button("Predict")

# load the pickle files
with open('notebook/pre.pkl', 'rb') as file1:
    pre = pickle.load(file1)

with open('notebook/model.pkl', 'rb') as file2:
    model = pickle.load(file2)

# logic for prediction
if submit:
    # convert the data into dataframe
    dct = {'sepal_length':[sep_len],
           'sepal_width':[sep_wid],
           'petal_length':[pet_len],
           'petal_width':[pet_wid]}
    # convert the  above dictionary to dataframe
    xnew = pd.DataFrame(dct)
    # transform xnew
    xnew_pre = pre.transform(xnew)
    # predict the result with probability
    pred = model.predict(xnew_pre)
    prob = model.predict_proba(xnew_pre)
    max_prob = np.max(prob)
    # print above result
    st.subheader('prediction are : ')
    st.subheader(f'Predicted Species : {pred[0]}')
    st.subheader(f'Probability : {max_prob*100:.2f} %')
    st.progress((max_prob))