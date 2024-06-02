import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Функция для тренировки модели логистической регрессии
def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Интерфейс для загрузки файла
st.title("Simple Logistic Regression App")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Чтение данных из загруженного файла
    data = pd.read_csv(uploaded_file)
    st.write("Dataset", data)
    
    # Выбор фичей и таргета
    features = st.multiselect("Select features for regression", data.columns)
    target = st.selectbox("Select target column", data.columns)
    
    if features and target:
        X = data[features]
        y = data[target]
        
        # Нормализация данных
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Тренировка модели
        model = train_model(X_scaled, y)
        
        # Вывод весов модели
        coef_dict = {feature: coef for feature, coef in zip(features, model.coef_[0])}
        st.write("Regression Coefficients", coef_dict)
        
        # Scatter plot
        feature_x = st.selectbox("Select feature for x-axis", features)
        feature_y = st.selectbox("Select feature for y-axis", features)
        
        if feature_x and feature_y:
            fig, ax = plt.subplots()
            scatter = ax.scatter(data[feature_x], data[feature_y], c=y, cmap='viridis')
            ax.set_xlabel(feature_x)
            ax.set_ylabel(feature_y)
            legend1 = ax.legend(*scatter.legend_elements(), title=target)
            ax.add_artist(legend1)
            st.pyplot(fig)