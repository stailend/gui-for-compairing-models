import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_selected_curves(model_data, selected_graphs, confusion_matrices):
    os.makedirs("results", exist_ok=True)

    # --- Confusion Matrix ---
    if "Confusion Matrix" in selected_graphs:
        for model_name, conf_matrix in confusion_matrices.items():
            plt.figure(figsize=(6, 5))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix ({model_name})")
            plt.savefig(f"results/conf_matrix_{model_name}.png")
            plt.show()

    # --- ROC-AUC Train ---
    if "ROC-AUC Train" in selected_graphs:
        plt.figure(figsize=(10, 5))
        for model_name, (y_train, y_proba_train, _, _) in model_data.items():
            fpr_train, tpr_train, _ = roc_curve(y_train, y_proba_train)
            plt.plot(fpr_train, tpr_train, label=f"{model_name} Train ROC")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid()
        plt.title("ROC-кривая (Train)")
        plt.get_current_fig_manager().set_window_title("ROC-кривая (Train)")
        plt.legend()
        plt.savefig("results/roc_train.png")
        plt.show()

    # --- ROC-AUC Test ---
    if "ROC-AUC Test" in selected_graphs:
        plt.figure(figsize=(10, 5))
        for model_name, (_, _, y_test, y_proba_test) in model_data.items():
            fpr_test, tpr_test, _ = roc_curve(y_test, y_proba_test)
            plt.plot(fpr_test, tpr_test, label=f"{model_name} Test ROC")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid()
        plt.title("ROC-кривая (Test)")
        plt.get_current_fig_manager().set_window_title("ROC-кривая (Test)")
        plt.legend()
        plt.savefig("results/roc_test.png")
        plt.show()

    # --- PR-AUC Train ---
    if "PR-AUC Train" in selected_graphs:
        plt.figure(figsize=(10, 5))
        for model_name, (y_train, y_proba_train, _, _) in model_data.items():
            precision_train, recall_train, _ = precision_recall_curve(y_train, y_proba_train)
            plt.plot(recall_train, precision_train, label=f"{model_name} Train PR")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid()
        plt.title("PR-кривая (Train)")
        plt.get_current_fig_manager().set_window_title("PR-кривая (Train)")
        plt.legend()
        plt.savefig("results/pr_train.png")
        plt.show()

    # --- PR-AUC Test ---
    if "PR-AUC Test" in selected_graphs:
        plt.figure(figsize=(10, 5))
        for model_name, (_, _, y_test, y_proba_test) in model_data.items():
            precision_test, recall_test, _ = precision_recall_curve(y_test, y_proba_test)
            plt.plot(recall_test, precision_test, label=f"{model_name} Test PR")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("PR-кривая (Test)")
        plt.grid()
        plt.get_current_fig_manager().set_window_title("PR-кривая (Test)")
        plt.legend()
        plt.savefig("results/pr_test.png")
        plt.show()