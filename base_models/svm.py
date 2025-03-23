from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import joblib
import os


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
            "PR-AUC": average_precision_score(y_test, y_proba_test)
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