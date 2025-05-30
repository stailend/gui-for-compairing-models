import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QCheckBox, QPushButton,
    QComboBox, QFormLayout, QLineEdit, QGroupBox, QHBoxLayout, QTextEdit, QGridLayout)
import heart


class MLInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML models analysis")
        self.setGeometry(100, 100, 900, 750)
        self.controller = heart.MLController()
        
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        dataset_label = QLabel("Choose dataset:")
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["UNSW-NB15"])
        self.dataset_combo.addItems(["Random"])


        self.models_group = QGroupBox("Pick models")
        model_layout = QGridLayout()
        self.models = {}

        model_names = ["CatBoost", "NN", "RandomForest", "KNN", "PWL", "CART", "SVM"]
        for index, model in enumerate(model_names):
            group_box = QGroupBox(model)
            group_layout = QFormLayout()
            enable_checkbox = QCheckBox("Enable")
            enable_checkbox.setChecked(False)

            params = {}
            if model == "CatBoost":
                params["Tree Depth"] = QLineEdit("6")
                params["Number of Iterations"] = QLineEdit("100")
                params["Learning Rate"] = QLineEdit("0.02")
            elif model == "NN":
                params["Hidden Layer Size"] = QLineEdit("64")
                params["Learning Rate"] = QLineEdit("0.01")
                params["Epochs"] = QLineEdit("100")
            elif model == "SVM":
                params["C Coefficient"] = QLineEdit("1.0")
                params["Kernel"] = QLineEdit("rbf")
            elif model == "RandomForest":
                params["Number of Trees"] = QLineEdit("50")
                params["Maximum Depth"] = QLineEdit("None")
            elif model == "KNN":
                params["Number of Neighbors"] = QLineEdit("4")
                params["Metric"] = QLineEdit("minkowski")
            elif model == "PWL":
                params["Learning Rate"] = QLineEdit("0.05")
                params["Epochs"] = QLineEdit("500")
            elif model == "CART":
                params["Maximum Depth"] = QLineEdit("None")
                params["Minimum Samples to Split"] = QLineEdit("2")
            group_layout.addRow(enable_checkbox)
            for param, field in params.items():
                group_layout.addRow(param, field)

            group_box.setLayout(group_layout)
            row, col = divmod(index, 3) 
            model_layout.addWidget(group_box, row, col)
            self.models[model] = {"enabled": enable_checkbox, "params": params}

        self.models_group.setLayout(model_layout)

        self.metrics_group = QGroupBox("Choose metrics")
        metrics_layout = QHBoxLayout()
        self.metrics = {m: QCheckBox(m) for m in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC", "Log-Loss"]}
        for checkbox in self.metrics.values():
            checkbox.setChecked(False)
            metrics_layout.addWidget(checkbox)

        self.metrics_group.setLayout(metrics_layout)

        self.graphs_group = QGroupBox("Choose graphs")
        graphs_layout = QHBoxLayout()
        self.graphs = {g: QCheckBox(g) for g in ["ROC-AUC Train", "ROC-AUC Test", "PR-AUC Train", "PR-AUC Test", "Confusion Matrix", "Classes"]}
        for checkbox in self.graphs.values():
            checkbox.setChecked(False)
            graphs_layout.addWidget(checkbox)

        self.graphs_group.setLayout(graphs_layout)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_training)


        
        self.results_box = QTextEdit()
        self.results_box.setReadOnly(True)

        layout.addWidget(dataset_label)
        layout.addWidget(self.dataset_combo)
        layout.addWidget(self.models_group)
        layout.addWidget(self.metrics_group)
        layout.addWidget(self.graphs_group)
        layout.addWidget(self.run_button)
        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.results_box)


        self.setLayout(layout)

    def run_training(self):
        selected_models = [m for m in self.models if self.models[m]["enabled"].isChecked()]
        model_params = {
            m: {p: self.models[m]["params"][p].text() for p in self.models[m]["params"]}
            for m in selected_models
        }
        selected_metrics = [m for m in self.metrics if self.metrics[m].isChecked()]
        selected_graphs = [g for g in self.graphs if self.graphs[g].isChecked()]
        
        dataset = self.dataset_combo.itemText(self.dataset_combo.currentIndex())
        self.controller.data(dataset)
        metrics = self.controller.train_and_evaluate(selected_models, model_params, selected_metrics, selected_graphs)

        result_text = "=== Final metrics ===\n"
        for model, data in metrics.items():
            result_text += f"\n[{model}]\n"
            for metric, value in data.items():
                result_text += f"{metric}: {value:.4f}\n"

        self.results_box.setText(result_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLInterface()
    window.show()
    sys.exit(app.exec())