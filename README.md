# MLOPs
# Leaf Classification Tool

This project provides a simple web application for classifying leaves based on their features. It utilizes Streamlit for the web interface and Joblib to load a pre-trained machine learning model.

## Features

- **Interactive Web Interface**: Built with Streamlit, allowing users to input leaf features and receive instant classification results.
- **Machine Learning Model**: Uses a Support Vector Machine (SVC) model trained on leaf features and saved with Joblib for quick loading and prediction.

## Technologies Used

- **Streamlit**: An open-source app framework for Machine Learning and Data Science projects.
- **Joblib**: Used for saving and loading the trained machine learning model efficiently.
- **Scikit-Learn**: For training the machine learning model.

## Start the Streamlit web app

streamlit run app.py

This command will start the Streamlit server and open your web browser to the address http://localhost:8501, where you can interact with the application.

Input Features: Users adjust sliders to set values for leaf features such as Eccentricity, Aspect Ratio, etc.
Model Prediction: The input features are processed by a pre-trained SVC model loaded through Joblib, which then predicts the leaf's class based on these features.
