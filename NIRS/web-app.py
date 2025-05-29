import streamlit as st
import pickle
import pandas as pd

# Загрузка модели
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Предсказание медицинских расходов")

age = st.slider("Возраст", 18, 100, 30)
sex = st.selectbox("Пол", ['male', 'female'])
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.slider("Количество детей", 0, 5, 0)
smoker = st.selectbox("Курит", ['yes', 'no'])
region = st.selectbox("Регион", ['northeast', 'northwest', 'southeast', 'southwest'])

input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

if st.button("Предсказать"):
    prediction = model.predict(input_data)
    st.write(f"Предполагаемые медицинские расходы: ${prediction[0]:.2f}")
