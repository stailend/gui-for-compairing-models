import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class NNModel:
    def __init__(self, input_size, hidden_size=64, learning_rate=0.01, epochs=10):
        self.model = SimpleNN(input_size, hidden_size)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.metrics = {}
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_train = self.scaler.fit_transform(X_train)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            self.optimizer.step()

    def evaluate(self, X_train, X_test, y_train, y_test):
        self.model.eval()  # Переключаем модель в режим тестирования
        with torch.no_grad():
            X_train = self.scaler.transform(X_train)
            X_test = self.scaler.transform(X_test)
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

            y_proba_train = torch.sigmoid(self.model(X_train_tensor)).detach().numpy()
            y_proba_test = torch.sigmoid(self.model(X_test_tensor)).detach().numpy()

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

    def save_model(self, path="models/nn_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="models/nn_model.pth"):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print("Модель NN загружена.")
        else:
            print("Файл модели не найден!")

    def plot_decision(self, X_test, y_test):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import numpy as np

        x_min, x_max = X_test[:, 0].min() - 0.1, X_test[:, 0].max() + 0.1
        y_min, y_max = X_test[:, 1].min() - 0.1, X_test[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_tensor = torch.tensor(grid, dtype=torch.float32)
        with torch.no_grad():
            Z = torch.sigmoid(self.model(grid_tensor)).numpy().reshape(xx.shape)
        Z = np.round(Z)

        plt.figure(figsize=(8, 6))
        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(['blue', 'red'])  
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=custom_cmap)        
        plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], c='blue', s=20, alpha=0.8, edgecolors='k', linewidths=0.2, label='Class 0 (Normal)')
        plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], c='red', s=20, alpha=0.8, edgecolors='k', linewidths=0.2, label='Class 1 (Anomaly, Rotated)')
        plt.xlabel("Признак 1")
        plt.ylabel("Признак 2")
        plt.title("Граница решения модели нейросети")
        plt.legend()
        plt.grid()
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()