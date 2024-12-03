import csv
import torch
import numpy as np


# Function to write metrics to CSV
def write_metrics_to_csv(metrics, csv_file):
    """
    Write training and validation metrics to a CSV file.

    Parameters:
    - metrics (list of dict): List of dictionaries containing epoch, train_loss, train_acc, val_loss, and val_acc.
    - csv_file (str): The name of the CSV file to write to.
    """
    if not metrics:
        return
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    print(f"Metrics written to {csv_file}")


# Function to load model weights
def load_model_weights(model, file_path, weights_only=True):
    model.load_state_dict(torch.load(file_path, weights_only=weights_only))
    print(f"Weights loaded from {file_path}")


def compute_accuracy_from_confusion_matrix(cm):
    # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
    true_positive = np.diag(cm)  # Diagonal represents correct predictions

    false_positive = cm.sum(axis=0) - true_positive  # FP: sum of columns - TP
    false_negative = cm.sum(axis=1) - true_positive  # FN: sum of rows - TP
    true_negative = cm.sum() - (
        true_positive + false_positive + false_negative
    )  # TN: total sum - TP - FP - FN

    # Accuracy: Total correct predictions / Total samples
    accuracy = true_positive.sum() / cm.sum()

    # Sensitivity (Recall) = TP / (TP + FN)
    sensitivity = true_positive / (true_positive + false_negative)

    # Specificity = TN / (TN + FP)
    specificity = true_negative / (true_negative + false_positive)

    # Balanced Accuracy: (Sensitivity + Specificity) / 2
    balanced_accuracy = (sensitivity + specificity) / 2

    return accuracy, balanced_accuracy.mean()
