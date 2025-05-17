from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, log_loss, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np


class SVMModel:
    def __init__(self, C=1.0, kernel="rbf"):
        self.model = SVC(C=C, kernel=kernel, probability=True)
        self.metrics = {}

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_train, X_test, y_train, y_test):
        y_proba_train = self.model.predict_proba(X_train)[:, 1]
        y_proba_test = self.model.predict_proba(X_test)[:, 1]

        y_pred_test = self.model.predict(X_test)

        self.metrics = {
            "Accuracy": accuracy_score(y_test, y_pred_test),
            "Precision": precision_score(y_test, y_pred_test, zero_division=0),
            "Recall": recall_score(y_test, y_pred_test),
            "F1-score": f1_score(y_test, y_pred_test),
            "ROC-AUC": roc_auc_score(y_test, y_proba_test),
            "PR-AUC": average_precision_score(y_test, y_proba_test),
            "Log-Loss": log_loss(y_test, y_proba_test)
        }

        return y_proba_train, y_proba_test, confusion_matrix(y_test, y_pred_test)

    def save_model(self, path="models/svm_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load_model(self, path="models/svm_model.pkl"):
        if os.path.exists(path):
            self.model = joblib.load(path)
            print("Модель SVM загружена.")
        else:
            print("Файл модели не найден!")
            
    def plot_decision(self, X_test, y_test):
        
        x_min, x_max = X_test[:, 0].min() - 0.1, X_test[:, 0].max() + 0.1
        y_min, y_max = X_test[:, 1].min() - 0.1, X_test[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.model.predict(grid)
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(8, 6))
        plt.xlabel("Признак 1")
        plt.ylabel("Признак 2")
        plt.title("Граница решения модели SVM")
        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(['blue', 'red'])  
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=custom_cmap)

        plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], c='blue', s=20, alpha=0.8, edgecolors='k', linewidths=0.2, label='Class 0 (Normal)')
        plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], c='red', s=20, alpha=0.8, edgecolors='k', linewidths=0.2, label='Class 1 (Anomaly, Rotated)')

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid()
        plt.show()