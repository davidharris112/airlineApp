# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# used chat gpt for general syntax & organizational help

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('Penguin Classifier: A Machine Learning App') 
st.write("This app uses 6 inputs to predict the species of penguin using a model "
         "built on the Palmer's Penguin's dataset. Use the inputs in the sidebar to "
         "make your prediction!")

# Display an image of penguins
st.image('penguins.png', width = 400)

# Load the pre-trained model from the pickle file
dt_pickle = open('decision_tree_penguin.pickle', 'rb') 
clf = pickle.load(dt_pickle) 
dt_pickle.close()


# Create a sidebar for input collection
st.sidebar.header('**Penguin Features Input**')

# Button to choose input type
input_type = st.sidebar.radio("Single Penguin or Multiple", ("Single Penguin", "Multiple") )

# If input type is Single Penguin, provide inputs for characteristics
if input_type=="Single Penguin":
    # Sidebar input fields for categorical variables
    island = st.sidebar.selectbox('Penguin Island', options = ['Biscoe', 'Dream', 'Torgerson'])
    sex = st.sidebar.selectbox('Sex', options = ['Female', 'Male'])

    # Sidebar input fields for numerical variables using sliders
    bill_length_mm = st.sidebar.slider('Bill Length (mm)', min_value=32.0, max_value=60.0, step=0.1)
    bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', min_value=13.0, max_value=21.0, step=0.1)
    flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', min_value=170.0, max_value=230.0, step=0.5)
    body_mass_g = st.sidebar.slider('Body Mass (g)', min_value=2700.0, max_value=6300.0, step=100.0)

    # Putting sex and island variables into the correct format
    # so that they can be used by the model for prediction
    island_Biscoe, island_Dream, island_Torgerson = 0, 0, 0 
    if island == 'Biscoe': 
        island_Biscoe = 1 
    elif island == 'Dream': 
        island_Dream = 1 
    elif island == 'Torgerson': 
        island_Torgerson = 1 

    sex_female, sex_male = 0, 0 
    if sex == 'Female': 
        sex_female = 1 
    elif sex == 'Male': 
        sex_male = 1 

    # Using predict() with new data provided by the user
    new_prediction = clf.predict([[bill_length_mm, bill_depth_mm, flipper_length_mm, 
    body_mass_g, island_Biscoe, island_Dream, island_Torgerson, sex_female, sex_male]]) 

    # Store the predicted species
    prediction_species = new_prediction[0]

    # Display input summary
    st.write("### Input Summary")
    st.write(f"**Island**: {island}")
    st.write(f"**Sex**: {sex}")
    st.write(f"**Bill Length**: {bill_length_mm} mm")
    st.write(f"**Bill Depth**: {bill_depth_mm} mm")
    st.write(f"**Flipper Length**: {flipper_length_mm} mm")
    st.write(f"**Body Mass**: {body_mass_g} g")

    # Show the predicted species on the app
    st.subheader("Predicting Your Penguin's Species")
    st.success(f'We predict your penguin is of the **{prediction_species}** species.')

# If input type is Multile, ask for CSV
if input_type == "Multiple":
    csv = st.sidebar.file_uploader("Upload CSV", type = ["csv"])
    if csv is not None:
        penguins_df = pd.read_csv(csv)

        # Ensure required columns exist
        required_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'island', 'sex']
        missing_cols = [col for col in required_cols if col not in penguins_df.columns]

        # Create Columns of zeros for one-hot encoding
        penguins_df["island_Biscoe"] = 0
        penguins_df["island_Dream"] = 0
        penguins_df["island_Torgersen"] = 0
        penguins_df["sex_female"] = 0
        penguins_df["sex_male"] = 0

        for index, row in penguins_df.iterrows():
            if row["island"] == 'Biscoe': 
                penguins_df.loc[index, 'island_Biscoe'] = 1 
            elif row["island"] == 'Dream': 
                penguins_df.loc[index, 'island_Dream'] = 1 
            elif row["island"] == 'Torgersen': 
                penguins_df.loc[index, 'island_Torgersen'] = 1 

            if row["sex"] == 'female': 
                penguins_df.loc[index, 'sex_female'] = 1 
            elif row["sex"] == 'male': 
                penguins_df.loc[index, 'sex_male'] = 1 
            bill_length_mm = penguins_df.loc[index, 'bill_length_mm']
            bill_depth_mm = penguins_df.loc[index, 'bill_depth_mm']
            flipper_length_mm = penguins_df.loc[index, 'flipper_length_mm']
            body_mass_g = penguins_df.loc[index, 'body_mass_g']
            island_Biscoe = penguins_df.loc[index, 'island_Biscoe']
            island_Dream = penguins_df.loc[index, 'island_Dream']
            island_Torgerson = penguins_df.loc[index, 'island_Torgersen']
            sex_female = penguins_df.loc[index, 'sex_female']
            sex_male = penguins_df.loc[index, 'sex_male']

            # Using predict() with new data provided by the csv
            new_prediction = clf.predict([[bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, island_Biscoe, island_Dream, island_Torgerson, sex_female, sex_male]]) 

            # Store the predicted species
            penguins_df.loc[index, "Predicted Species"] = new_prediction[0]



        with st.expander("Predicted Species"):
            display_df = penguins_df[['species', 'island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year', 'Predicted Species']]    
            st.dataframe(display_df, width = "stretch")
    else:
        st.info("Please upload a CSV file to get predictions for multiple penguins.")


# Showing Feature Importance plot
st.write('We used a machine learning model **(Decision Tree)** to predict the species. '
         'The features used in this prediction are ranked by relative importance below.')
st.image('feature_imp.svg')


#----------------------------------------------------------
# Showing additional items in tabs
st.subheader("Prediction Performance")
tab1, tab2, tab3, tab4 = st.tabs(["Decision Tree", "Feature Importance", "Confusion Matrix", "Classification Report"])

# Tab 1: Visualizing Decision Tree
with tab1:
    st.write("### Decision Tree Visualization")
    st.image('dt_visual.svg')
    st.caption("Visualization of the Decision Tree used in prediction.")

# Tab 2: Feature Importance Visualization
with tab2:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Features used in this prediction are ranked by relative importance.")

# Tab 3: Confusion Matrix
with tab3:
    st.write("### Confusion Matrix")
    st.image('confusion_mat.svg')
    st.caption("Confusion Matrix of model predictions.")

# Tab 4: Classification Report
with tab4:
    st.write("### Classification Report")
    report_df = pd.read_csv('class_report.csv', index_col = 0).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
    st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")
