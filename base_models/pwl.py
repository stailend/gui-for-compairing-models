import pandas as pd
import os
import numpy as np
import joblib
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.special import expit as sigmoid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import torch

import_indexes = [ 'ct_srv_src', 'dmean',
 'ct_srv_dst', 'smean', 'sbytes', 'ct_state_ttl', 'sttl']

class pwlModel:
    def __init__(self, input_dim, learning_rate=0.01, epochs=500, pca_components=2):
        self.input_dim = pca_components + 1
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights1 = torch.tensor(np.random.rand(self.input_dim), dtype=torch.float32, requires_grad=True)
        self.weights2 = torch.tensor(np.random.rand(self.input_dim), dtype=torch.float32, requires_grad=True)
        self.optimizer = optim.Adam([self.weights1, self.weights2], lr=self.learning_rate)
        self.scaler = StandardScaler()
        self.metrics = {}

    def r_func(self, f1, f2):
        return f1 + f2 + np.sqrt(f1**2 + f2**2)

    def multiplyer(self, net1, net2):
        return 1 - net2 / np.sqrt(net1**2 + net2**2)

    def sigmoid_der(self, x):
        return sigmoid(x) * (1 - sigmoid(x))

    def train(self, X_train, y_train):
        print(type(X_train))
        X_train = self.scaler.fit_transform(X_train)
        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train)) 

        for epoch in range(self.epochs):
            net1 = np.dot(X_train, self.weights1.detach().numpy())
            net2 = np.dot(X_train, self.weights2.detach().numpy())

            net = self.r_func(net1, net2)
            predictions = np.round(sigmoid(net))
            errors = y_train - predictions

            grad_w1 = np.dot(errors * self.sigmoid_der(net1) * self.multiplyer(net1, net2), X_train) / X_train.shape[0]
            grad_w2 = np.dot(errors * self.sigmoid_der(net2) * self.multiplyer(net2, net1), X_train) / X_train.shape[0]

            lambda_reg = 0.01

            grad_w1 = torch.tensor(grad_w1, dtype=torch.float32)
            grad_w2 = torch.tensor(grad_w2, dtype=torch.float32)

            self.weights1 = self.weights1 + self.learning_rate * (grad_w1 - lambda_reg * self.weights1)
            self.weights2 = self.weights2 + self.learning_rate * (grad_w2 - lambda_reg * self.weights2)

            loss = torch.tensor(np.mean(errors**2), requires_grad=True) 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, X_train, X_test, y_train, y_test):
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)
        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

        y_proba_train = sigmoid(self.r_func(np.dot(X_train, self.weights1.detach().numpy()), np.dot(X_train, self.weights2.detach().numpy())))
        y_proba_test = sigmoid(self.r_func(np.dot(X_test, self.weights1.detach().numpy()), np.dot(X_test, self.weights2.detach().numpy())))

        y_pred_test = np.round(y_proba_test)

        self.metrics = {
            "Accuracy": accuracy_score(y_test, y_pred_test),
            "Precision": precision_score(y_test, y_pred_test, zero_division=0),
            "Recall": recall_score(y_test, y_pred_test),
            "F1-score": f1_score(y_test, y_pred_test),
            "ROC-AUC": roc_auc_score(y_test, y_proba_test),
            "PR-AUC": average_precision_score(y_test, y_proba_test)
        }

        return y_proba_train, y_proba_test, confusion_matrix(y_test, y_pred_test)

    def save_model(self, path="models/PWL_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"weights1": self.weights1.detach().numpy(), "weights2": self.weights2.detach().numpy(), "scaler": self.scaler, "pca": self.pca}, path)

    def load_model(self, path="models/PWL_model.pkl"):
        if os.path.exists(path):
            data = joblib.load(path)
            self.weights1 = torch.tensor(data["weights1"], dtype=torch.float32, requires_grad=True)
            self.weights2 = torch.tensor(data["weights2"], dtype=torch.float32, requires_grad=True)
            self.scaler = data["scaler"]
            self.pca = data["pca"]
            self.optimizer = optim.Adam([self.weights1, self.weights2], lr=self.learning_rate)
            print("Модель PWL загружена.")
        else:
            print("Файл модели не найден!")

    def plot_decision(self, X_test, y_test, ):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import numpy as np

        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))

        x_min, x_max = X_test_scaled[:, 1].min() - 3.1, X_test_scaled[:, 1].max() + 3.1
        y_min, y_max = X_test_scaled[:, 2].min() - 3.1, X_test_scaled[:, 2].max() + 3.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))

        grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
        net1 = np.dot(grid, self.weights1.detach().numpy())
        net2 = np.dot(grid, self.weights2.detach().numpy())
        Z = np.round(sigmoid(self.r_func(net1, net2)).reshape(xx.shape))



        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        plt.scatter(X_test_scaled[y_test == 0][:, 1], X_test_scaled[y_test == 0][:, 2], color='red', label='Класс 0')
        plt.scatter(X_test_scaled[y_test == 1][:, 1], X_test_scaled[y_test == 1][:, 2], color='green', label='Класс 1')
        plt.legend()
        plt.xlabel("Признак 1")
        plt.ylabel("Признак 2")
        plt.title("Граница решения модели PWL")
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.grid(True)
        plt.show()
