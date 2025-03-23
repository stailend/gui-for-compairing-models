import pandas as pd
import os
from sklearn.decomposition import PCA

class DatasetLoader:
    def __init__(self, train_path="data/UNSW_NB15_training-set.csv", test_path="data/UNSW_NB15_testing-set.csv"):
        self.train_path = train_path
        self.test_path = test_path
        self.features = None 
        self.target_name = "label"
        self.train_data = None
        self.test_data = None
        self.pca = PCA(n_components=2)


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

        X_train = self.pca.fit_transform(X_train)
        X_test = self.pca.transform(X_test)


        return X_train, X_test, y_train, y_test, self.features, self.target_name


if __name__ == "__main__":
    loader = DatasetLoader()
    loader.load_and_preprocess()
    X_train, X_test, y_train, y_test, features, target = loader.get_data()
    
    print(f"Обучающая выборка: {X_train.shape}, Тестовая выборка: {X_test.shape}")
