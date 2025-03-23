import pandas as pd
import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.special import expit as sigmoid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

class pwlModel:
    def __init__(self, input_dim = 39, learning_rate=0.01, epochs=500):
        """
        Инициализация модели pwl-классификатора.
        :param input_dim: Количество входных признаков.
        :param learning_rate: Скорость обучения.
        :param epochs: Количество эпох.
        """
        self.input_dim = input_dim + 1  # Добавляем bias
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights1 = np.random.rand(self.input_dim)
        self.weights2 = np.random.rand(self.input_dim)
        self.scaler = StandardScaler()
        self.metrics = {}

    def r_func(self, f1, f2):
        """Функция R-оператора."""
        return f1 + f2 + np.sqrt(f1**2 + f2**2)

    def multiplyer(self, net1, net2):
        """Функция нормализации."""
        return 1 - net2 / np.sqrt(net1**2 + net2**2)

    def sigmoid_der(self, x):
        """Производная сигмоиды."""
        return sigmoid(x) * (1 - sigmoid(x))

    def train(self, X_train, y_train):
        """
        Обучает модель pwl-классификации.
        """
        X_train = self.scaler.fit_transform(X_train)
        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))  # Добавляем bias

        for epoch in range(self.epochs):
            net1 = np.dot(X_train, self.weights1)
            net2 = np.dot(X_train, self.weights2)

            net = self.r_func(net1, net2)
            predictions = np.round(sigmoid(net))
            errors = y_train - predictions

            grad_w1 = np.dot(errors * self.sigmoid_der(net1) * self.multiplyer(net1, net2), X_train) / X_train.shape[0]
            grad_w2 = np.dot(errors * self.sigmoid_der(net2) * self.multiplyer(net2, net1), X_train) / X_train.shape[0]

            self.weights1 += self.learning_rate * grad_w1
            self.weights2 += self.learning_rate * grad_w2

    def evaluate(self, X_train, X_test, y_train, y_test):
        """
        Оценивает модель на тестовых данных и считает метрики.
        """
        y_proba_train = self.predict_proba(X_train)
        y_proba_test = self.predict_proba(X_test)
        y_pred_test = self.predict(X_test)

        self.metrics = {
            "Accuracy": accuracy_score(y_test, y_pred_test),
            "Precision": precision_score(y_test, y_pred_test, zero_division=0),
            "Recall": recall_score(y_test, y_pred_test),
            "F1-score": f1_score(y_test, y_pred_test),
            "ROC-AUC": roc_auc_score(y_test, y_proba_test),
            "PR-AUC": average_precision_score(y_test, y_proba_test)
        }

        return y_proba_train, y_proba_test, confusion_matrix(y_test, y_pred_test)

    def predict_proba(self, X):
        """
        Возвращает вероятности предсказаний для входных данных X.
        """
        X = self.scaler.transform(X)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return sigmoid(self.r_func(np.dot(X, self.weights1), np.dot(X, self.weights2)))

    def predict(self, X):
        """
        Возвращает бинарные предсказания (0 или 1).
        """
        return np.round(self.predict_proba(X))

    def save_model(self, path="models/pwl_model.pkl"):
        """
        Сохраняет модель в файл.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"weights1": self.weights1, "weights2": self.weights2, "scaler": self.scaler}, path)

    def load_model(self, path="models/pwl_model.pkl"):
        """
        Загружает модель из файла.
        """
        if os.path.exists(path):
            data = joblib.load(path)
            self.weights1 = data["weights1"]
            self.weights2 = data["weights2"]
            self.scaler = data["scaler"]
            print("Модель pwl загружена.")
        else:
            print("Файл модели не найден!")

