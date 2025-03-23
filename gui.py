import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QCheckBox, QPushButton,
    QComboBox, QFormLayout, QLineEdit, QGroupBox, QHBoxLayout, QTextEdit, QGridLayout
)
import heart


class MLInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система анализа моделей ML")
        self.setGeometry(100, 100, 900, 750)
        self.controller = heart.MLController()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Логгирование в GUI
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log("Интерфейс загружен.")

        # Выбор датасета
        dataset_label = QLabel("Выберите датасет:")
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["UNSW-NB15"])

        # Блок моделей
        self.models_group = QGroupBox("Выбор моделей")
        model_layout = QGridLayout()
        self.models = {}

        model_names = ["CatBoost", "NN", "SVM", "RandomForest", "KNN", "PWL", "CART"]
        for index, model in enumerate(model_names):
            group_box = QGroupBox(model)
            group_layout = QFormLayout()
            enable_checkbox = QCheckBox("Включить")
            enable_checkbox.setChecked(False)

            params = {}
            if model == "CatBoost":
                params["Глубина деревьев"] = QLineEdit("6")
                params["Количество итераций"] = QLineEdit("100")
                params["Скорость обучения"] = QLineEdit("0.05")
            elif model == "NN":
                params["Размер скрытого слоя"] = QLineEdit("128")
                params["Скорость обучения"] = QLineEdit("0.01")
                params["Эпохи"] = QLineEdit("100")
            elif model == "SVM":
                params["Коэффициент C"] = QLineEdit("1.0")
                params["Ядро"] = QLineEdit("rbf")
            elif model == "RandomForest":
                params["Количество деревьев"] = QLineEdit("100")
                params["Максимальная глубина"] = QLineEdit("None")
            elif model == "KNN":
                params["Количество соседей"] = QLineEdit("5")
                params["Метрика"] = QLineEdit("minkowski")
            elif model == "PWL":
                params["Скорость обучения"] = QLineEdit("0.01")
                params["Эпохи"] = QLineEdit("50")
            elif model == "CART":
                params["Максимальная глубина"] = QLineEdit("None")
                params["Минимальное количество для разбиения"] = QLineEdit("2")

            group_layout.addRow(enable_checkbox)
            for param, field in params.items():
                group_layout.addRow(param, field)

            group_box.setLayout(group_layout)
            row, col = divmod(index, 3) 
            model_layout.addWidget(group_box, row, col)
            self.models[model] = {"enabled": enable_checkbox, "params": params}

        self.models_group.setLayout(model_layout)

        self.metrics_group = QGroupBox("Выбор метрик")
        metrics_layout = QHBoxLayout()
        self.metrics = {m: QCheckBox(m) for m in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"]}
        for checkbox in self.metrics.values():
            checkbox.setChecked(False)
            metrics_layout.addWidget(checkbox)

        self.metrics_group.setLayout(metrics_layout)

        self.graphs_group = QGroupBox("Выбор графиков")
        graphs_layout = QHBoxLayout()
        self.graphs = {g: QCheckBox(g) for g in ["ROC-AUC Train", "ROC-AUC Test", "PR-AUC Train", "PR-AUC Test", "Confusion Matrix", "Classes"]}
        for checkbox in self.graphs.values():
            checkbox.setChecked(False)
            graphs_layout.addWidget(checkbox)

        self.graphs_group.setLayout(graphs_layout)

        self.run_button = QPushButton("Запустить обучение")
        self.run_button.clicked.connect(self.run_training)

        self.results_box = QTextEdit()
        self.results_box.setReadOnly(True)

        layout.addWidget(dataset_label)
        layout.addWidget(self.dataset_combo)
        layout.addWidget(self.models_group)
        layout.addWidget(self.metrics_group)
        layout.addWidget(self.graphs_group)
        layout.addWidget(self.run_button)
        layout.addWidget(QLabel("Результаты:"))
        layout.addWidget(self.results_box)
        layout.addWidget(QLabel("Лог:"))
        layout.addWidget(self.log_box)

        self.setLayout(layout)

    def log(self, message):
        """Выводит логи в интерфейс."""
        self.log_box.append(message)

    def run_training(self):
        self.log("Запуск обучения...")
        selected_models = [m for m in self.models if self.models[m]["enabled"].isChecked()]
        model_params = {
            m: {p: self.models[m]["params"][p].text() for p in self.models[m]["params"]}
            for m in selected_models
        }
        selected_metrics = [m for m in self.metrics if self.metrics[m].isChecked()]
        selected_graphs = [g for g in self.graphs if self.graphs[g].isChecked()]

        self.log(f"Выбранные модели: {selected_models}")
        self.log(f"Выбранные метрики: {selected_metrics}")
        self.log(f"Выбранные графики: {selected_graphs}")

        metrics = self.controller.train_and_evaluate(selected_models, model_params, selected_metrics, selected_graphs)

        result_text = "=== Итоговые метрики ===\n"
        for model, data in metrics.items():
            result_text += f"\n[{model}]\n"
            for metric, value in data.items():
                result_text += f"{metric}: {value:.4f}\n"

        self.results_box.setText(result_text)
        self.log("Обучение завершено.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLInterface()
    window.show()
    sys.exit(app.exec())