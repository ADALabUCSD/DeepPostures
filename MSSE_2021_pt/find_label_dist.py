import os
import pickle
import torch
from commons import get_dataloaders

# Set data path to Hip or Wrist
data_path = "/niddk-data-central/iWatch/pre_processed_pt/W"  # change to W if needed

# Load subject splits
with open("/niddk-data-central/iWatch/support_files/iwatch_split_dict.pkl", "rb") as f:
    split_data = pickle.load(f)

train_subjects = split_data["train"]
valid_subjects = None

# Get dataloader
data_loader_train, _, _ = get_dataloaders(
    pre_processed_dir=data_path,
    bi_lstm_win_size=42,
    batch_size=32,
    train_subjects=train_subjects,
    valid_subjects=valid_subjects,
    test_subjects=None,
)

# Count positives and negatives
positive_count = 0
negative_count = 0

for inputs, labels in data_loader_train:
    labels = labels.view(-1)
    positive_count += (labels == 1).sum().item()  # non-sitting
    negative_count += (labels == 0).sum().item()  # sitting

# Compute pos_weight
if positive_count == 0:
    raise ValueError("No positive samples found in training data")
pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)

print(f"positive_count: {positive_count}")
print(f"negative_count: {negative_count}")
print(f"pos_weight: {pos_weight}")


'''
result:
Hip:
positive_count: 456636
negative_count: 1276452
pos_weight: tensor([2.7953])

Wrist:
positive_count: 467112
negative_count: 1318728
pos_weight: tensor([2.8232])




'''