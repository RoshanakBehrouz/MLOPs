import streamlit as st
from joblib import load
import numpy as np

# Load the trained model
model = load('model.joblib')

# Streamlit interface
st.title('Leaf Identification')

# Display a picture in the sidebar
st.sidebar.image('Leaf.png', caption='Leaf Identification', use_column_width=True)

# Define the feature names and their expected ranges
features = {
    "Eccentricity": (0.0, 1.0),
    "Aspect Ratio": (1.0, 20.0),
    "Elongation": (0.0, 1.0),
    "Solidity": (0.4, 1.0),
    "Stochastic Convexity": (0.3, 1.0),
    "Isoperimetric Factor": (0.0, 1.0),
    "Maximal Indentation Depth": (0.0, 0.2),
    "Lobedness": (0.0, 8.0),
    "Average Intensity": (0.0, 0.2),
    "Average Contrast": (0.0, 0.3),
    "Smoothness": (0.0, 0.08),
    "Third Moment": (0.0, 0.03),
    "Uniformity": (0.0, 0.003),
    "Entropy": (0.0, 3.0)
}

# Arrange sliders in rows with three columns each
inputs = []
feature_items = list(features.items())
rows = (len(features) + 2) // 3  # Calculate number of rows needed

for _ in range(rows):
    cols = st.columns(3)
    for col in cols:
        if feature_items:
            feature, (min_val, max_val) = feature_items.pop(0)
            input_val = col.slider(f'Enter {feature}', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2.0, step=(max_val - min_val) / 100)
            inputs.append(input_val)

if st.button('Predict'):
    inputs = np.array(inputs).reshape(1, -1)
    prediction = model.predict(inputs)
    st.write(f'Predicted Class: {prediction[0]}')

from joblib import load
import numpy as np

# Load the trained model
model = load('model.joblib')

# Streamlit interface
st.title('Leaf Species Identification Manual Insertion')

# Input fields for features
inputs = []
for feature in ["Eccentricity", "Aspect Ratio", "Elongation", "Solidity", "Stochastic Convexity", "Isoperimetric Factor", "Maximal Indentation Depth", "Lobedness", "Average Intensity", "Average Contrast", "Smoothness", "Third Moment", "Uniformity", "Entropy"]:
    inputs.append(st.number_input(f'Enter {feature}', format="%.5f"))

if st.button('*Predict*'):
    inputss = np.array(inputs).reshape(1, -1)
    predictions = model.predict(inputss)
    st.write(f'Predicted Class: {predictions[0]}')



import streamlit as st
import pandas as pd

# Load the dataset and create a mapping from class numbers to scientific names
data = pd.read_csv('leaf.csv')
class_to_name = data.set_index('Class (Species)')['Leaf Scientific name'].to_dict()

# Setting up the Streamlit interface
st.title('Convert class number to Leaf Scientific Name')

# User input for class number
class_input = st.number_input('Enter Leaf Class Number', min_value=1, value=1, step=1)

# Button to retrieve and display the scientific name
if st.button('Get Scientific Name'):
    scientific_name = class_to_name.get(class_input, "Scientific name not found.")
    st.write(f'The scientific name for class {class_input} is: {scientific_name}')




