import dataset
import results

from base_models import model_catboost
from base_models import nn
from base_models import svm
from base_models import random_forest
from base_models import knn
from base_models import pwl
from base_models import cart
import numpy as np

class MLController:
    def __init__(self):
        self.dataset_loader = dataset.DatasetLoader()
        self.dataset_loader.load_and_preprocess()
        self.X_train, self.X_test, self.y_train, self.y_test, self.features, self.target = self.dataset_loader.get_data()

    def train_and_evaluate(self, selected_models, model_params, selected_metrics, selected_graphs):
        results_data = {}
        model_data = {}
        confusion_matrices = {}
        indexes = np.random.randint(0, self.X_test.shape[0], 1000)


        for model_name in selected_models:
            if model_name == "CatBoost":
                model = model_catboost.CatBoostModel(
                    depth=int(model_params[model_name]["Глубина деревьев"]),
                    iterations=int(model_params[model_name]["Количество итераций"]),
                    learning_rate=float(model_params[model_name]["Скорость обучения"])
                )
            elif model_name == "NN":
                model = nn.NNModel(
                    input_size=self.X_train.shape[1],
                    hidden_size=int(model_params[model_name]["Размер скрытого слоя"]),
                    learning_rate=float(model_params[model_name]["Скорость обучения"]),
                    epochs=int(model_params[model_name]["Эпохи"])
                )
            elif model_name == "SVM":
                model = svm.SVMModel(
                    C=float(model_params[model_name]["Коэффициент C"]),
                    kernel=model_params[model_name]["Ядро"]
                )
            elif model_name == "RandomForest":
                model = random_forest.RandomForestModel(
                    n_estimators=int(model_params[model_name]["Количество деревьев"]),
                    max_depth=int(model_params[model_name]["Максимальная глубина"]) if model_params[model_name]["Максимальная глубина"] != "None" else None
                )
            elif model_name == "KNN":
                model = knn.KNNModel(
                    n_neighbors=int(model_params[model_name]["Количество соседей"]),
                    metric=model_params[model_name]["Метрика"]
                )
            elif model_name == "PWL":
                model = pwl.pwlModel(
                    input_dim=self.X_train.shape[1],
                    learning_rate=float(model_params[model_name]["Скорость обучения"]),
                    epochs=int(model_params[model_name]["Эпохи"])
                )
            elif model_name == "CART":
                model = cart.cartModel(
                    max_depth=int(model_params[model_name]["Максимальная глубина"]) if model_params[model_name]["Максимальная глубина"] != "None" else None,
                    min_samples_split=int(model_params[model_name]["Минимальное количество для разбиения"])
                )
            model.train(self.X_train, self.y_train)
            y_proba_train, y_proba_test, conf_matrix = model.evaluate(self.X_train, self.X_test, self.y_train, self.y_test)
            if "Classes" in selected_graphs:
                self.example_X_test = self.X_test[indexes]
                self.example_y_test = self.y_test[indexes]
                model.plot_decision(self.example_X_test, self.example_y_test)
            
            #model.save_model()
            #if model_name == "CatBoost":
            #    model.plot_feature_importance(self.features)

            results_data[model_name] = {metric: model.metrics[metric] for metric in selected_metrics}
            model_data[model_name] = (self.y_train, y_proba_train, self.y_test, y_proba_test)
            confusion_matrices[model_name] = conf_matrix

        results.plot_selected_curves(model_data, selected_graphs, confusion_matrices)

        

        return results_data