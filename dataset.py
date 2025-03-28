import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

class DatasetLoader:
    def __init__(self, train_path="data/UNSW_NB15_training-set.csv", test_path="data/UNSW_NB15_testing-set.csv"):
        self.train_path = train_path
        self.test_path = test_path
        self.features = None 
        self.target_name = "label"
        self.train_data = None
        self.test_data = None


    def load_and_preprocess(self):
        self.train_data = self._read_and_clean(self.train_path)
        self.test_data = self._read_and_clean(self.test_path)

        if self.train_data is None or self.test_data is None:
            raise ValueError("Ошибка при загрузке датасета!")

        self.features = [col for col in self.train_data.columns if col != self.target_name]

    def _read_and_clean(self, file_path):
        if not os.path.exists(file_path):
            print(f"Файл {file_path} не найден!")
            return None

        df = pd.read_csv(file_path)

        if "id" in df.columns:
            df.drop(columns=["id"], inplace=True)

        for col in df.columns:
            if col == self.target_name:
                continue 
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                df.drop(columns=[col], inplace=True)

        return df


    def get_data(self):
        if self.train_data is None or self.test_data is None:
            raise ValueError("Данные не загружены! Вызовите load_and_preprocess().")
        
        X_train = self.train_data[self.features]
        y_train = self.train_data[self.target_name]

        X_test = self.test_data[self.features]
        y_test = self.test_data[self.target_name]

        return X_train, X_test, y_train, y_test, self.features, self.target_name


if __name__ == "__main__":
    loader = DatasetLoader()
    loader.load_and_preprocess()
    X_train, X_test, y_train, y_test, features, target = loader.get_data()
    
    print(f"Обучающая выборка: {X_train.shape}, Тестовая выборка: {X_test.shape}")
    
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    import matplotlib.pyplot as plt

    # Масштабирование данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучаем модель случайного леса для оценки важности признаков
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Получаем индексы двух наиболее важных признаков
    importances = rf.feature_importances_
    top2_indices = np.argsort(importances)[-2:]
    top2_features = [features[i] for i in top2_indices]

    # Оставляем только два признака
    X_train_top2 = X_train[top2_features]
    X_test_top2 = X_test[top2_features]


    scaler = StandardScaler()
    X_train_top2 = scaler.fit_transform(X_train_top2)
    X_test_top2 = scaler.transform(X_test_top2)


    # Визуализация
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_test_top2[:, 0], X_test_top2[:, 1], c=y_test, cmap="viridis", alpha=0.6)
    plt.colorbar(scatter, label='Класс')
    plt.title(f"Визуализация по двум наиболее важным признакам: {top2_features[0]} и {top2_features[1]}")
    plt.xlabel(top2_features[0])
    plt.ylabel(top2_features[1])
    plt.grid(True)
    plt.show()
