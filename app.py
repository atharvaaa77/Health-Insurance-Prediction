import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Importing the pickle File :
load_model= pickle.load(open('health_claim.pkl', 'rb'))
data = pd.read_csv('insurance.csv')

# Creating the app layouts
st.markdown("-------------------------------------------------------------------------------------------------------")
st.title("HEALTH INSURANCE PREDICTION ")
st.image("health-insurance-concept.png")
st.markdown("-------------------------------------------------------------------------------------------------------")

nav = st.sidebar.radio("Navigation" ,["Home" ,"Prediction"])
if nav == 'Home':
    if st.checkbox("DISPLAY TABLE"):
        st.markdown("DISPLAYING THE TABLE ")
        st.dataframe(data)

    st.markdown("----------------------------------------------------------------------------------------------------")

if nav == 'Prediction':
    st.write("Calculating the INSURANCE CHARGES that could be charged by an Insurer based on a Person's Attributes")
    st.markdown("---------------------------------------------------------------------------------------------------")

    def load_data():
        df = pd.DataFrame({'sex': ['Male', 'Female'],
                           'smoker': ['Yes', 'No']})
        return df
    df = load_data()

    def load_data():
        df1 = pd.DataFrame({'region': ['Southeast', 'Northwest', 'Southwest', 'Northeast']})
        return df1
    df1 = load_data()
    sex = st.radio ("Select Gender :", df['sex'].unique())
    age = st.number_input(" Enter your Age :", 18, 65)
    region = st.selectbox("Select Region :", df1['region'].unique())
    smoker = st.radio("Are you a smoker ?", df['smoker'].unique())
    children = st.number_input("Number of children", 0, 5)
    bmi = st.slider(" Enter your Body Mass Index :", 15, 55)

    if sex == 'male':
        gender = 1
    else:
        gender = 0

    if smoker == 'yes':
        smoke = 1
    else:
        smoke = 0

    if region == 'Northeast':
        reg = 0
    elif region == 'Northwest':
        reg = 1
    elif region == 'Southeast':
        reg = 2
    else:
        reg = 3

    features = [gender, age, reg, smoke, bmi, children]

    # convert user inputs into an array for the model
    int_features = [int(x) for x in features]
    final_features = [np.array(int_features)]

    if st.button('PREDICT YOUR RESULT'):
        prediction = load_model.predict(final_features)
        st.success(f'HEALTH INSURANCE COST: ₹ {round(prediction[0], 2)}')

    st.markdown("----------------------------------------------------------------------------------------------------")


