
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from math import radians, cos, sin


class DatasetLoader:
    def __init__(self):#, train_path="data/UNSW_NB15_training-set.csv", test_path="data/UNSW_NB15_testing-set.csv"):
        pass

    def load_and_preprocess(self):
        pass
    def get_data(self):
        np.random.seed(42)
        n_blue = 7500
        n_red = 4500

        x0 = np.random.normal(loc=0, scale=0.6, size=(n_blue, 2))
        x0[:, 1] += 1.3 
        x1 = np.random.normal(loc=0, scale=0.8, size=(n_red, 2))
        x1[:, 1] = 0.5 * np.abs(x1[:, 0]) + np.random.normal(loc=-0.5, scale=0.3, size=n_red)
        angle_deg = 10
        theta = radians(angle_deg)
        rotation_matrix = np.array([
            [cos(theta), sin(theta)],
            [-sin(theta), cos(theta)]
        ])
        x1_rotated = x1 @ rotation_matrix.T

        X = np.vstack((x0, x1_rotated))
        y = np.hstack((np.zeros(n_blue), np.ones(n_red)))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        self.features = ['x0', 'x1']
        self.target_name = 'y'
        return X_train, X_test, y_train, y_test, self.features, self.target_name
