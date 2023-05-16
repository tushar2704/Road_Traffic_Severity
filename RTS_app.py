#RTS Application by Tushar Aggarwal

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prediction import get_prediction, ordinal_encoder
st.set_page_config(page_title="Road Accident Severity Prediction App",
                   page_icon="🛣️", layout="wide")


# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

model = joblib.load(r'notebooks\random_forest_final.joblib')


# Title of application
st.title("🛣️Road Accident Severity Prediction App🛣️")
st.markdown( '### by Tushar Aggarwal')

st.markdown("""This app predict the Severity of the accident from the selected attributes, **Data Source:**  [Kaggle](https://www.kaggle.com/datasets/tusharaggarwal27/rtadata1).
""")

#creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']
       
options_cause = ['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly',
       'No priority to vehicle', 'Moving Backward',
       'No priority to pedestrian', 'Other', 'Overtaking',
       'Driving under the influence of drugs', 'Driving to the left',
       'Getting off the vehicle improperly', 'Driving at high speed',
       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Unknown', 'Improper parking']
options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
       'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
       'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
       'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']
options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']
options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
       'other', 'Double carriageway (median)', 'One way',
       'Two-way (divided with solid lines road marking)', 'Unknown']

features = ['hour','day_of_week','casualties','accident_cause','vehicles_involved','vehicle_type','driver_age','accident_area','driving_experience','lanes']

#Line


def main():
    with st.form('prediction_form'):

        st.sidebar.subheader("Enter the input for following features:")
        
        hour = st.sidebar.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
        day_of_week = st.sidebar.selectbox("Select Day of the Week: ", options=options_day)
        casualties = st.sidebar.slider("Hour of Accident: ", 1, 8, value=0, format="%d")
        accident_cause = st.sidebar.selectbox("Select Accident Cause: ", options=options_cause)
        vehicles_involved = st.sidebar.slider("Pickup Hour: ", 1, 7, value=0, format="%d")
        vehicle_type = st.sidebar.selectbox("Select Vehicle Type: ", options=options_vehicle_type)
        driver_age = st.sidebar.selectbox("Select Driver Age: ", options=options_age)
        accident_area = st.sidebar.selectbox("Select Accident Area: ", options=options_acc_area)
        driving_experience = st.sidebar.selectbox("Select Driving Experience: ", options=options_driver_exp)
        lanes = st.sidebar.selectbox("Select Lanes: ", options=options_lanes)
    

        st.write("Your selection are as follows: ")
        st.write("Hour: ", hour,", Day: ", day_of_week,", Casualties: ", casualties,
                  ", Accident Cause: ", accident_cause,", Vehicle Involved: ", vehicles_involved,
                  ", Vehicle Type: ", vehicle_type,", Driver Age: ", driver_age,
                  ", Accident Area: ", accident_area,
                  ", Driving Experience: ", driving_experience)
      
        submit = st.form_submit_button("Predict")

        st.markdown("<hr style='border: 2px solid #85C1E9;'>", unsafe_allow_html=True)

    if submit:
        
        day_of_week = ordinal_encoder(day_of_week, options_day)
        accident_cause = ordinal_encoder(accident_cause, options_cause)
        vehicle_type = ordinal_encoder(vehicle_type, options_vehicle_type)
        driver_age =  ordinal_encoder(driver_age, options_age)
        accident_area =  ordinal_encoder(accident_area, options_acc_area)
        driving_experience = ordinal_encoder(driving_experience, options_driver_exp) 
        lanes = ordinal_encoder(lanes, options_lanes)


        data = np.array([hour,day_of_week,casualties,accident_cause,vehicles_involved, 
                            vehicle_type,driver_age,accident_area,driving_experience,lanes]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)

        st.markdown(f"### The predicted severity is:  {pred[0]}")
st.markdown("<hr style='border: 2px solid #85C1E9;'>", unsafe_allow_html=True)
if __name__ == '__main__':
    main()