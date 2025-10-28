# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

st.title('Penguin Classifier: A Machine Learning App') 

# Display the image
st.image('penguins.png', width = 400)

st.write("This app uses 6 inputs to predict the species of penguin using " 
         "a model built on the Palmer's Penguin's dataset. Use the following form or upload your dataset" 
         " to get started!") 

# Load the pre-trained model from the pickle file
dt_pickle = open('decision_tree_penguin.pickle', 'rb') 
clf = pickle.load(dt_pickle) 
dt_pickle.close()

# Create a sidebar for input collection
st.sidebar.header('Penguin Features Input')

# Option 1: Asking users to input their data as a file
penguin_file = st.sidebar.file_uploader('Option 1: Upload your own penguin data')

# Option 2: Asking users to input their data using a form in the sidebar
st.sidebar.write('Option 2: Use the following form')

#---------------------------------------------------------------------------------------------
# Using Default (Original) Dataset to Automate Few Items
#---------------------------------------------------------------------------------------------

# Load the default dataset
default_df = pd.read_csv('penguins.csv')
default_df = default_df.dropna().reset_index(drop = True) 
# NOTE: drop = True is used to avoid adding a new column for old index

# For categorical variables, using selectbox
island = st.sidebar.selectbox('Penguin Island', options = default_df['island'].unique()) 
sex = st.sidebar.selectbox('Sex', options = default_df['sex'].unique()) 

# Sidebar input fields for numerical variables using sliders
# NOTE: Make sure that variable names are same as that of training dataset
bill_length_mm = st.sidebar.slider('Bill Length (mm)', 
                                   min_value = default_df['bill_length_mm'].min(), 
                                   max_value = default_df['bill_length_mm'].max(), 
                                   step = 0.1)

bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', 
                                  min_value = default_df['bill_depth_mm'].min(), 
                                  max_value = default_df['bill_depth_mm'].max(), 
                                  step = 0.1)

flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', 
                                      min_value = default_df['flipper_length_mm'].min(), 
                                      max_value = default_df['flipper_length_mm'].max(), 
                                      step = 0.5)

body_mass_g = st.sidebar.slider('Body Mass (g)', 
                                min_value = default_df['body_mass_g'].min(), 
                                max_value = default_df['body_mass_g'].max(), 
                                step = 100.0)

# # Putting sex and island variables into the correct format
# # so that they can be used by the model for prediction
# island_Biscoe, island_Dream, island_Torgerson = 0, 0, 0 
# if island == 'Biscoe': 
#    island_Biscoe = 1 
# elif island == 'Dream': 
#    island_Dream = 1 
# elif island == 'Torgerson': 
#    island_Torgerson = 1 

# sex_female, sex_male = 0, 0 
# if sex == 'Female': 
#    sex_female = 1 
# elif sex == 'Male': 
#    sex_male = 1 


# If no file is provided, then allow user to provide inputs using the form
if penguin_file is None:
    # Encode the inputs for model prediction
    encode_df = default_df.copy()
    encode_df = encode_df.drop(columns = ['species', 'year'])

    # Combine the list of user data as a row to default_df
    encode_df.loc[len(encode_df)] = [island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(1)

    # Using predict() with new data provided by the user
    new_prediction = clf.predict(user_encoded_df)

    # Show the predicted species on the app
    st.subheader("Predicting Your Penguin's Species")
    st.success('**We predict your penguin is of the {} species**'.format(new_prediction[0])) 

else:
   # Loading data
   user_df = pd.read_csv(penguin_file) # User provided data
   original_df = pd.read_csv('penguins.csv') # Original data to create ML model
   
   # Dropping null values
   user_df = user_df.dropna().reset_index(drop = True) 
   original_df = original_df.dropna().reset_index(drop = True)
   
   # Remove output (species) and year columns from original data
   original_df = original_df.drop(columns = ['species', 'year'])
   # Remove year column from user data
   user_df = user_df.drop(columns = ['year'])
   
   # Ensure the order of columns in user data is in the same order as that of original data
   user_df = user_df[original_df.columns]

   # Concatenate two dataframes together along rows (axis = 0)
   combined_df = pd.concat([original_df, user_df], axis = 0)

   # Number of rows in original dataframe
   original_rows = original_df.shape[0]

   # Create dummies for the combined dataframe
   combined_df_encoded = pd.get_dummies(combined_df)

   # Split data into original and user dataframes using row index
   original_df_encoded = combined_df_encoded[:original_rows]
   user_df_encoded = combined_df_encoded[original_rows:]

   # Predictions for user data
   user_pred = clf.predict(user_df_encoded)

   # Predicted species
   user_pred_species = user_pred

   # Adding predicted species to user dataframe
   user_df['Predicted Species'] = user_pred_species
   
   # Show the predicted species on the app
   st.subheader("Predicting Your Penguin's Species")
   st.dataframe(user_df)

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
