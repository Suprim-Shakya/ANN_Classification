import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Loading the trained model
model = tf.keras.models.load_model('regression_model.h5')

# Loading the encoder and scaler
with open('label_encoder.pkl','rb') as f:
    label_encoder = pickle.load(f)
with open('onehot_encoder.pkl','rb') as f:
    onehot = pickle.load(f)
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

st.title("Estimated Salary Prediction")

## Input fields
geography = st.selectbox('Geography', onehot.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.number_input('Exited', 0, 1, 0)
tenure = st.number_input('Tenure', 0, 10, 5)
num_of_products = st.number_input('Number of Products', 1, 4, 2)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Creating a DataFrame for the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' : [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited],
})

# One-hot encoding the Geography column
Geography_encoded = onehot.transform([[geography]]).toarray()
Geography_encoded_df = pd.DataFrame(Geography_encoded, columns=onehot.get_feature_names_out(['Geography']))

# Dropping the original Geography column and concatenating the encoded columns
input_data = pd.concat([input_data.reset_index(drop=True), Geography_encoded_df], axis=1)

# Scaling the input data
input_data_scaled = scaler.transform(input_data)

# Making the prediction
prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]

st.write(f"Predicted Estimated Salary: {predicted_salary:.2f}")
st.write("Note: The predicted salary is based on the input features provided.")