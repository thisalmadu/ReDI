import numpy as np
import pandas as pd
import pickle
import streamlit as st
from PIL import Image



pickle_in_1 = open("xgboost_class.pkl", "rb")
classifier_1 = pickle.load(pickle_in_1)

def welcome():
    return "Welcome All"

def model_predict(pclass, sibl, parch, fare, gender, embarked):

    # Prediction of one record => Goes in the deplopyment
    best_model = classifier_1.best_estimator_

    # Data, going to make the prediction on
    new_data = {
        'PassengerId' : int(pid),
        'Pclass' : int(pclass),
        'Sex' : gender,
        'SibSp': int(sibl),
        'Parch' : int(parch),
        'Fare' : float(fare),
        'Embarked': embarked
    }

    # Preprocess the categorical data
    encoded_data = best_model.named_steps['encoder'].transform([new_data])

    # Make predictions using the best model
    prediction = best_model.predict(encoded_data)
    return prediction

def main():

    st.title("Titatic catastropy survival")
    html_temp = """
    <dev style = "background-color:tomato; padding:10px">
    <h2 style = "color:white;text-align:center;"> Will you survive from Titanic catatrophy? </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    pid = st.text_input("Passenger ID", "Enter the pasenger ID")
    pclass = st.selectbox('Passenger class',(1, 2, 3))
    sibl = st.text_input("Sibling count", "Number of siblings he/she have")
    parch = st.text_input("parent count", "Number of parents he/she have")
    fare = st.text_input("Passenger_fare", "fare chaeged from passenger")
    sex = st.selectbox('Gender',("male","female"))
    embark = st.selectbox('Port of embarken',("C", "Q", "S"))
    result = ""
    status = ""

    if st.button("Predict"):
        result = model_predict(pclass, sibl, parch, fare, sex, embark)
        if result == 1:
            status = "survive"
        else:
            status = "does not survived"
    
    st.success('This passenger {}'.format(status))


if __name__=='__main__':
        main()
