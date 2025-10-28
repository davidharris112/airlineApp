import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Get path relative to current script
BASE_DIR = os.path.dirname(__file__)
pickle_path = os.path.join(BASE_DIR, "decision_tree_airline.pickle")

# Load the classifier
with open(pickle_path, "rb") as dt_pickle:
#dt_pickle = open('decision_tree_airline.pickle', 'rb') 
    clf = pickle.load(dt_pickle) 
dt_pickle.close()
default_df = pd.read_csv('airline.csv')
default_df = default_df.dropna().reset_index(drop = True)

# Set up the title and description of the app
st.title("Airline Customer Satisfaction")
st.subheader("This app predicts airline customer satisfaction using a Decision Tree model")
st.image('airline.jpg')


st.sidebar.subheader("Customer Details")
with st.sidebar:
    customer_type = st.pills("Customer Type", ["Loyal Customer", "disloyal Customer"], selection_mode="single")
    type_of_travel = st.pills("Type of Travel", ["Personal Travel", "Business travel"], selection_mode="single")
    class_type = st.pills("Class", ["Eco", "Eco Plus", "Business"], selection_mode="single")
    age = st.number_input("Customer Age", min_value=0, max_value=120)

st.sidebar.subheader("Flight Details")
with st.sidebar:
    flight_distance = st.number_input("Flight Distance (in miles)", min_value=0)
    departure_arrival_time_convenient = st.radio("Departure/Arrival Time Convenience", [1, 2, 3, 4, 5], horizontal=True)
    departure_delay_in_minutes = st.number_input("Departure Delay (minutes)", min_value=0)
    arrival_delay_in_minutes = st.number_input("Arrival Delay (minutes)", min_value=0)

st.sidebar.subheader("Experience Ratings")
with st.sidebar:
    seat_comfort = st.radio("Seat Comfort", [1, 2, 3, 4, 5], horizontal=True)
    food_and_drink = st.radio("Food & Drink", [1, 2, 3, 4, 5], horizontal=True)
    gate_location = st.radio("Gate Location", [1, 2, 3, 4, 5], horizontal=True)
    inflight_wifi_service = st.radio("Inflight WiFi Service", [1, 2, 3, 4, 5], horizontal=True)
    inflight_entertainment = st.radio("Inflight Entertainment", [1, 2, 3, 4, 5], horizontal=True)
    online_support = st.radio("Online Support", [1, 2, 3, 4, 5], horizontal=True)
    ease_of_online_booking = st.radio("Ease of Online Booking", [1, 2, 3, 4, 5], horizontal=True)
    on_board_service = st.radio("On-board Service", [1, 2, 3, 4, 5], horizontal=True)
    leg_room_service = st.radio("Leg Room Service", [1, 2, 3, 4, 5], horizontal=True)
    baggage_handling = st.radio("Baggage Handling", [1, 2, 3, 4, 5], horizontal=True)
    checkin_service = st.radio("Check-in Service", [1, 2, 3, 4, 5], horizontal=True)
    cleanliness = st.radio("Cleanliness", [1, 2, 3, 4, 5], horizontal=True)
    online_boarding = st.radio("Online Boarding", [1, 2, 3, 4, 5], horizontal=True)




if st.sidebar.button("Predict"):

    # Encode the inputs for model prediction
    encode_df = default_df.copy()
    encode_df = encode_df.drop(columns = ['satisfaction'])

    # All user input data
    features = [customer_type, age, type_of_travel, class_type, flight_distance, seat_comfort, departure_arrival_time_convenient, food_and_drink, gate_location, inflight_wifi_service, inflight_entertainment, online_support, ease_of_online_booking, on_board_service, leg_room_service, baggage_handling, checkin_service, cleanliness, online_boarding, departure_delay_in_minutes, arrival_delay_in_minutes]

    # Combine the list of user data as a row to default_df
    encode_df.loc[len(encode_df)] = features

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(1)

    # Using predict() with new data provided by the user
    new_prediction = clf.predict(user_encoded_df)


# get probability
    # Store the predicted species
    prediction_class = new_prediction[0]

    # Predict class probabilities
    proba = clf.predict_proba(user_encoded_df)

    # Get probability of the predicted class
    class_index = list(clf.classes_).index(prediction_class)
    predicted_class_proba = proba[class_index]


    # Show the predicted species on the app
    st.subheader("Predicting The Customer's Satisfaction")
    st.success(f'We predict your customer is **{new_prediction[0]}**.'
               f'(Probability: {predicted_class_proba[0]:.2f})')
    
    st.subheader("Customer Demographic Details")
    st.write(f"- **Age:** {age}")
    st.write(f"- **Type of Travel:** {type_of_travel}")
    st.write(f"- **Class:** {class_type}")


    # Showing additional items in tabs
    st.subheader("Prediction Performance")
    tab1, tab2, tab3, tab4 = st.tabs(["Decision Tree", "Feature Importance", "Confusion Matrix", "Classification Report"])

    # Tab 1: Visualizing Decision Tree
    with tab1:
            st.image('dt_visual.svg')
            st.write("### Decision Tree Visualization")
            st.caption("Visualization of the Decision Tree used in prediction.")


    # Tab 2: Feature Importance Visualization
    with tab2:
        st.write("### Feature Importance")
        st.image('feature_imp.svg')


    # Tab 3: Confusion Matrix
    with tab3:
        st.write("### Confusion Matrix")
        st.image('confusion_mat.svg')    


    # Tab 4: Classification Report
    with tab4:
        st.write("### Classification Report")
        report_df = pd.read_csv('class_report.csv', index_col = 0).transpose()

else:
    st.info("ℹ️ Please fill out the survey form in the sidebar and click **Predict** to see the satisfaction prediction.")
