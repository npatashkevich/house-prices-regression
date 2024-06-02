import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Класс логистической регрессии
class LogReg:
    def __init__(self, learning_rate, n_epochs=10000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)  # Переведем в numpy для матричных преобразований
        y = np.array(y)

        self.coef_ = np.random.uniform(-1, 1, size=X.shape[1])
        self.intercept_ = np.random.uniform(-1, 1)

        for epoch in range(self.n_epochs):
            # Вычисляем линейную комбинацию входов и весов
            z = X @ self.coef_ + self.intercept_

            # Применяем сигмоидную функцию
            y_pred = self.sigmoid(z)

            # Вычисляем градиенты
            w0_grad = (y_pred - y)
            w_grad = X * (y_pred - y).reshape(-1, 1)

            # Обновляем параметры, используя коэффициент скорости обучения
            self.coef_ -= self.learning_rate * w_grad.mean(axis=0)
            self.intercept_ -= self.learning_rate * w0_grad.mean()

    def predict(self, X):
        y_pred = self.sigmoid(X @ self.coef_ + self.intercept_)
        return y_pred

    def score(self, X, y):
        y_pred = np.round(self.predict(X))
        return (y == y_pred).mean()

# Функция для тренировки модели
def train_model(X, y, learning_rate, n_epochs):
    model = LogReg(learning_rate=learning_rate, n_epochs=n_epochs)
    model.fit(X, y)
    return model

# Интерфейс для загрузки файла
st.title("Logistic Regression")

uploaded_file = st.file_uploader("Upload a CSV file for training", type="csv")

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

        # Параметры модели
        learning_rate = st.number_input("Learning Rate", value=0.01, step=0.01)
        n_epochs = st.number_input("Number of Epochs", value=1000, step=100)

        # Нормализация данных
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Тренировка модели
        model = train_model(X_scaled, y, learning_rate, n_epochs)
        
        # Вывод весов модели
        coef_str = " + ".join([f"{np.round(coef, 4)} * {feature}" for coef, feature in zip(model.coef_, features)])
        st.write(f'Personal.Loan = {coef_str} + {np.round(model.intercept_, 4)}')

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

        # Загрузка тестового датасета
        uploaded_test_file = st.file_uploader("Upload a CSV file for testing", type="csv")
        if uploaded_test_file:
            test_data = pd.read_csv(uploaded_test_file)
            st.write("Test Dataset", test_data)

            if all(feature in test_data.columns for feature in features):
                X_test = test_data[features]
                y_test = test_data[target]

                # Нормализация тестовых данных
                X_test_scaled = scaler.transform(X_test)

                # Расчет точности модели
                accuracy_train = model.score(X_scaled, y)
                accuracy_test = model.score(X_test_scaled, y_test)
                st.write(f'Accuracy train-set: {accuracy_train}')
                st.write(f'Accuracy test-set: {accuracy_test}')
            else:
                st.write("Test dataset does not contain all the required features.")

            # Вместо загрузки тестового датасета
            prediction_data = st.text_area("Enter data for prediction (each row as comma-separated values):")

            if prediction_data:
                # Преобразование введенных данных в датафрейм
                prediction_lines = prediction_data.strip().split('\n')
                prediction_values = [list(map(float, line.split(','))) for line in prediction_lines]
                prediction_df = pd.DataFrame(prediction_values, columns=X.columns)
                
                # Нормализация предсказательных данных
                X_pred_scaled = scaler.transform(prediction_df)
                
                # Получение предсказаний
                predictions = model.predict(X_pred_scaled)
                
                # Вывод предсказаний
                st.write("Predictions:")
                st.write(predictions)
