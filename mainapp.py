import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
import joblib 
import pickle

st.title("Diabetes Prediction Web App")

# loading the saved model

loaded_model=pickle.load(open("C:/Users/Dan Masibo/Desktop/diabetes_2/GradientBoost_1.pkl", 'rb'))

# creating a function for prediction

def diabetes_prediction(input_data):

    # changing input data as numpy array

    input_data_as_numpy_array=np.asarray(input_data)

    # reshape the array as we are predicting for one instance

    input_data_reshaped= input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return "This person is not diabetic"
    
    else:
        return "This person is diabetic"



def main():
     # giving a title

    st.title("Diabetes Pediction web App")


    # getting the input data from the user

    Pregnancies= st.text_input("Number of Pregnacies")
    Glucose= st.text_input("Glucose level")
    BloodPressure= st.text_input("Blood Pressure Value")
    SkinThickness= st.text_input("Skin Thickness Value")
    Insulin= st.text_input("Insulin Level")
    BMI= st.text_input("BMI Value")
    DiabetesPedigreeFunction= st.text_input("Diabetes Pedigree Function Value")
    Age= st.text_input("Age of the Person")

    # code for prediction

    diagnosis= ''

    # creating a button for prediction

    if st.button("Diabetes Test Result"):
        diagnosis= diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagnosis)

if __name__== '__main__':
    main()


