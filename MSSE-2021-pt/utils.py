import csv
import torch
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
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    
    print(f"Metrics written to {csv_file}")

# Function to load model weights
def load_model_weights(model, file_path, weights_only=True):
    model.load_state_dict(torch.load(file_path, weights_only=weights_only))
    print(f"Weights loaded from {file_path}")
