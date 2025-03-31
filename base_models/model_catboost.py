import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    average_precision_score, confusion_matrix
)

class CatBoostModel:
    def __init__(self, depth=6, iterations=1000, learning_rate=0.1):
        self.model = CatBoostClassifier(
            depth=depth,
            iterations=iterations,
            learning_rate=learning_rate,
            verbose=False
        )
        self.metrics = {}
        self.conf_matrix = None

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_train, X_test, y_train, y_test):
        y_pred_test = self.model.predict(X_test)
        y_proba_train = self.model.predict_proba(X_train)[:, 1]
        y_proba_test = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "Accuracy": accuracy_score(y_test, y_pred_test),
            "Precision": precision_score(y_test, y_pred_test, zero_division=0),
            "Recall": recall_score(y_test, y_pred_test),
            "F1-score": f1_score(y_test, y_pred_test),
            "ROC-AUC": roc_auc_score(y_test, y_proba_test),
            "PR-AUC": average_precision_score(y_test, y_proba_test)
        }

        self.conf_matrix = confusion_matrix(y_test, y_pred_test)

        return y_proba_train, y_proba_test, self.conf_matrix

    def plot_feature_importance(self, feature_names, save_path="results/catboost_feature_importance.png"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        importance = self.model.get_feature_importance()
        sorted_idx = np.argsort(importance)
        #print(np.array(feature_names)[sorted_idx][-5:])
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_names)), importance[sorted_idx], align="center")
        plt.yticks(range(len(feature_names)), np.array(feature_names)[sorted_idx])
        plt.xlabel("Feature Importance")
        plt.title("CatBoost Feature Importance")
        plt.savefig(save_path)
        plt.show()

    def save_model(self, path="models/catboost_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def plot_decision(self, X_test, y_test ):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import numpy as np

       

        x_min, x_max = X_test[:, 0].min() - 0.1, X_test[:, 0].max() + 0.1
        y_min, y_max = X_test[:, 1].min() - 0.1, X_test[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.model.predict(grid)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(['blue', 'red'])  
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=custom_cmap)        
        plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], c='blue', s=20, alpha=0.8, edgecolors='k', linewidths=0.2, label='Class 0 (Normal)')
        plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], c='red', s=20, alpha=0.8, edgecolors='k', linewidths=0.2, label='Class 1 (Anomaly, Rotated)')
        plt.xlabel("Признак 1")
        plt.ylabel("Признак 2")
        plt.title("Граница решения модели CatBoost")
        plt.legend()
        plt.grid()
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()