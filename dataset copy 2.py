import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

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
            raise ValueError("Dataset loading error!")

        self.features = [col for col in self.train_data.columns if col != self.target_name]

    def _read_and_clean(self, file_path):
        if not os.path.exists(file_path):
            print(f"File {file_path} not found!")
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

        X_train = self.train_data[self.features]
        y_train = self.train_data[self.target_name]

        X_test = self.test_data[self.features]
        y_test = self.test_data[self.target_name]

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.decomposition import PCA

        lda = LDA(n_components=1)
        pca = PCA(n_components=1)

        X_lda = lda.fit_transform(X_train, y_train)
        X_pca = pca.fit_transform(X_train)
        X_train_combined = np.hstack([X_lda, X_pca])
        
        X_lda = lda.fit_transform(X_test, y_test)
        X_pca = pca.fit_transform(X_test)
        X_test_combined = np.hstack([X_lda, X_pca])

        


        return X_train_combined, X_test_combined, y_train, y_test, self.features, self.target_name


if __name__ == "__main__":
    loader = DatasetLoader()
    loader.load_and_preprocess()
    X_train, X_test, y_train, y_test, features, target = loader.get_data()
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")



    indexes = np.random.randint(0, X_test.shape[0], 2000)
    X_test = X_test.iloc[indexes]
    y_test = y_test.iloc[indexes]

    
    """
    lda = LDA(n_components=1)
    X_test_lda = lda.fit_transform(X_test, y_test)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_test_lda, y_test, c = y_test, cmap="viridis", alpha=0.6)
    plt.colorbar(scatter, label='Класс')
    plt.title("LDA визуализация тестовой выборки")
    plt.xlabel("Компонента 1")
    plt.ylabel("Компонента 2")
    plt.grid(True)
    plt.show()
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    from sklearn.decomposition import PCA

    lda = LDA(n_components=1)
    X_lda = lda.fit_transform(X_test, y_test)

    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_test)

    X_combined = np.hstack([X_lda, X_pca])

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_combined[:, 0], X_combined[:, 1], c=y_test, cmap="viridis", alpha=0.6)
    plt.colorbar(scatter, label='Класс')
    plt.title("1D LDA + 1D PCA проекция")
    plt.xlabel("Компонента 1 (LDA)")
    plt.ylabel("Компонента 2 (PCA)")
    plt.grid(True)
    plt.show()