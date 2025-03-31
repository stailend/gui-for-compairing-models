from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

class cartModel:
    def __init__(self, max_depth=None, min_samples_split=2, pca_components=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.scaler = StandardScaler()
        self.model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        self.metrics = {}

    def train(self, X_train, y_train):
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def evaluate(self, X_train, X_test, y_train, y_test):
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)

        y_proba_train = self.model.predict_proba(X_train)[:, 1]
        y_proba_test = self.model.predict_proba(X_test)[:, 1]
        y_pred_test = self.model.predict(X_test)

        self.metrics = {
            "Accuracy": accuracy_score(y_test, y_pred_test),
            "Precision": precision_score(y_test, y_pred_test, zero_division=0),
            "Recall": recall_score(y_test, y_pred_test),
            "F1-score": f1_score(y_test, y_pred_test),
            "ROC-AUC": roc_auc_score(y_test, y_proba_test),
            "PR-AUC": average_precision_score(y_test, y_proba_test)
        }

        #self.plot_decision(X_test, y_test)


        return y_proba_train, y_proba_test, confusion_matrix(y_test, y_pred_test)

    def save_model(self, path="models/CART_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler }, path)

    def load_model(self, path="models/CART_model.pkl"):
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data["model"]
            self.scaler = data["scaler"]
            print("Модель CART загружена.")
        else:
            print("Файл модели не найден!")

    def plot_decision(self, X_test, y_test):
        import matplotlib.pyplot as plt
        import numpy as np

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
        plt.title("Граница решения модели CART")
        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(['blue', 'red'])  
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=custom_cmap)

        plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], c='blue', s=20, alpha=0.8, edgecolors='k', linewidths=0.2, label='Class 0 (Normal)')
        plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], c='red', s=20, alpha=0.8, edgecolors='k', linewidths=0.2, label='Class 1 (Anomaly, Rotated)')

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid()
        plt.show()