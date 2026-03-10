import streamlit as st
import pandas as pd
import joblib

kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title='Customer segmentation', page_icon='👥')
st.title('Customer Segmentation Predictor')

age = st.number_input('Age', value=35)
income = st.number_input('Income', value=50000)
total_spending = st.number_input('Total Spending', value=500)
web = st.number_input('Web Purchases', value=5)
store = st.number_input('Store Purchases', value=5)
recency = st.number_input('Recency', value=30)
children = st.number_input('Children', value=0)

input_data = pd.DataFrame([[age, income, total_spending, web, store, recency, children]], 
                         columns=['Age', 'Income', 'Total_Spending', 'NumWebPurchases', 
                                  'NumStorePurchases', 'Recency', 'Children'])
if st.button('Predict Segment'):
    scaled_input = scaler.transform(input_data)
    prediction = int(kmeans.predict(scaled_input)[0])

    cluster_names = {
        0: 'Budget Starters',
        1: 'High-Value Elites',
        2: 'Established Digital Shoppers',
        3: 'Family-Focused Seniors'
    }

    if prediction > 3:
        prediction = 3 

    result = cluster_names[prediction]
    st.success(f'Predicted Segment: {result}')