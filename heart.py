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
        self.X_train, self.X_test, self.y_train, self.y_test, self.features, self.target = 0,0,0,0,0,0
    def data(self, dataset = 'UNSW-NB15'):
        if dataset == 'UNSW-NB15':
            print(dataset)
            import datasets.UNSW_NB15 as dataset
            self.dataset_loader = dataset.DatasetLoader()
            self.dataset_loader.load_and_preprocess()
            self.X_train, self.X_test, self.y_train, self.y_test, self.features, self.target = self.dataset_loader.get_data()

        if dataset == 'Random':  
            print(dataset)
            import datasets.random as dataset
            self.dataset_loader = dataset.DatasetLoader()
            self.dataset_loader.load_and_preprocess()
            self.X_train, self.X_test, self.y_train, self.y_test, self.features, self.target = self.dataset_loader.get_data()

        else:
            print('unknown dataset')

    def train_and_evaluate(self, selected_models, model_params, selected_metrics, selected_graphs):
        results_data = {}
        model_data = {}
        confusion_matrices = {}

        for model_name in selected_models:
            if model_name == "CatBoost":
                model = model_catboost.CatBoostModel(
                    depth=int(model_params[model_name]["Tree Depth"]),
                    iterations=int(model_params[model_name]["Number of Iterations"]),
                    learning_rate=float(model_params[model_name]["Learning Rate"])
                )
            elif model_name == "NN":
                model = nn.NNModel(
                    input_size=self.X_train.shape[1],
                    hidden_size=int(model_params[model_name]["Hidden Layer Size"]),
                    learning_rate=float(model_params[model_name]["Learning Rate"]),
                    epochs=int(model_params[model_name]["Epochs"])
                )
            elif model_name == "SVM":
                model = svm.SVMModel(
                    C=float(model_params[model_name]["C Coefficient"]),
                    kernel=model_params[model_name]["Kernel"]
                )
            elif model_name == "RandomForest":
                model = random_forest.RandomForestModel(
                    n_estimators=int(model_params[model_name]["Number of Trees"]),
                    max_depth=int(model_params[model_name]["Maximum Depth"]) if model_params[model_name]["Maximum Depth"] != "None" else None
                )
            elif model_name == "KNN":
                model = knn.KNNModel(
                    n_neighbors=int(model_params[model_name]["Number of Neighbors"]),
                    metric=model_params[model_name]["Metric"]
                )
            elif model_name == "PWL":
                model = pwl.pwlModel(
                    input_dim=self.X_train.shape[1],
                    learning_rate=float(model_params[model_name]["Learning Rate"]),
                    epochs=int(model_params[model_name]["Epochs"])
                )
            elif model_name == "CART":
                model = cart.cartModel(
                    max_depth=int(model_params[model_name]["Maximum Depth"]) if model_params[model_name]["Maximum Depth"] != "None" else None,
                    min_samples_split=int(model_params[model_name]["Minimum Samples to Split"])
                )

            model.train(self.X_train, self.y_train)
            y_proba_train, y_proba_test, conf_matrix = model.evaluate(self.X_train, self.X_test, self.y_train, self.y_test)
            if "Classes" in selected_graphs:

                model.plot_decision(self.X_test, self.y_test)
            
            #model.save_model()
            #if model_name == "CatBoost":
            #    model.plot_feature_importance(self.features)

            results_data[model_name] = {metric: model.metrics[metric] for metric in selected_metrics}
            model_data[model_name] = (self.y_train, y_proba_train, self.y_test, y_proba_test)
            confusion_matrices[model_name] = conf_matrix

        results.plot_selected_curves(model_data, selected_graphs, confusion_matrices)

        return results_data