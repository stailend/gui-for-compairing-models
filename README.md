# ML Model Comparison GUI

This project provides a user-friendly graphical interface for configuring, training, and comparing various machine learning models on the UNSW-NB15 dataset. The application is built using **PyQt6** and is designed to assist users in evaluating classification models in the context of information security tasks.

## Features

- Interactive selection of models (e.g., CatBoost, Neural Network, SVM, Random Forest, KNN, PWL, CART)
- Configurable hyperparameters for each model
- Selection of evaluation metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC)
- Visualization options (ROC/PR curves, confusion matrix, class distribution)
- Real-time logging and result display

## Installation

1. Clone the repository:

```bash
git clone https://github.com/stailend/gui-for-compairing-models.git
cd gui-for-compairing-models
```
2.	Install dependencies:
```bash
pip install -r requirements.txt
```
3.	Run the application:
```bash
python gui.py
```

Requirements
	•	Python 3.8+
	•	PyQt6
	•	Other libraries used for modeling and visualization (defined in heart.py and requirements.txt)

Usage
	1.	Launch the GUI.
	2.	Select the dataset (currently only UNSW-NB15 is available).
	3.	Choose one or more models and adjust their parameters.
	4.	Select evaluation metrics and graphs to visualize.
	5.	Click Run Training to start training and see the results.

License

This project is open-source and licensed under the MIT License.
