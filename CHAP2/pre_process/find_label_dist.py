import argparse
import os
import pickle
from datetime import datetime, timedelta

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


# Adapted from the SOL_Finetune MSSE_2021_pt/find_label_dist.py utility.
# This script only uses the training split because pos_weight is a training loss parameter.
class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator


def input_iterator(data_root, subject_id, train=False):
    subject_dir = os.path.join(data_root, subject_id)
    fnames = [
        name.split(".")[0]
        for name in os.listdir(subject_dir)
        if not name.startswith(".") and name.endswith(".h5")
    ]
    fnames.sort()

    for i in range(len(fnames) - 1):
        assert datetime.strptime(fnames[i + 1], "%Y-%m-%d").date() - datetime.strptime(
            fnames[i], "%Y-%m-%d"
        ).date() == timedelta(days=1)

    data_batch = []
    label_batch = []

    for fname in fnames:
        with h5py.File(os.path.join(subject_dir, f"{fname}.h5"), "r") as h5f:
            data = h5f["data"][:]
            sleeping = h5f["sleeping"][:]
            non_wear = h5f["non_wear"][:]
            label = h5f["label"][:]

        for d, s, nw, l in zip(data, sleeping, non_wear, label):
            if s == 1 or nw == 1 or (train and l == -1):
                if len(label_batch) > 0:
                    yield np.array(data_batch), np.array(label_batch)
                data_batch = []
                label_batch = []
                continue

            data_batch.append(d)
            label_batch.append(l)

    if len(label_batch) > 0:
        yield np.array(data_batch), np.array(label_batch)


def window_generator(data_root, win_size_10s, subject_ids):
    for subject_id in subject_ids:
        subject_dir = os.path.join(data_root, subject_id)
        if not os.path.isdir(subject_dir):
            print(f"Subject data at {subject_dir} not found")
            continue

        for x_seq, y_seq in input_iterator(data_root, subject_id, train=True):
            x_window = []
            y_window = []
            for x, y in zip(x_seq, y_seq):
                x_window.append(x)
                y_window.append(y)

                if len(y_window) == win_size_10s:
                    yield np.stack(x_window, axis=0), np.stack(y_window, axis=0)
                    x_window = []
                    y_window = []


def get_train_dataloader(data_path, split_path, bi_lstm_win_size, batch_size):
    with open(split_path, "rb") as f:
        split_data = pickle.load(f)

    train_subjects = split_data["train"]
    train_data = IterDataset(
        window_generator(data_path, bi_lstm_win_size, train_subjects)
    )
    return DataLoader(train_data, batch_size=batch_size, pin_memory=True)


def compute_pos_weight(data_loader_train):
    # CHAP2 binary labels: 0 = sitting, 1 = non-sitting.
    positive_count = 0
    negative_count = 0

    for _, labels in data_loader_train:
        labels = labels.view(-1)
        positive_count += (labels == 1).sum().item()
        negative_count += (labels == 0).sum().item()

    if positive_count == 0:
        raise ValueError("No positive samples found in training data.")

    return positive_count, negative_count, negative_count / positive_count


def main():
    parser = argparse.ArgumentParser(
        description="Compute training-set label distribution and BCE pos_weight."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the preprocessed Hip or Wrist directory, e.g. /path/to/W.",
    )
    parser.add_argument(
        "--split-path",
        required=True,
        help="Path to the pickle split dictionary containing the 'train' subject list.",
    )
    parser.add_argument("--bi-lstm-win-size", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_loader_train = get_train_dataloader(
        args.data_path,
        args.split_path,
        args.bi_lstm_win_size,
        args.batch_size,
    )
    positive_count, negative_count, pos_weight = compute_pos_weight(data_loader_train)

    print(f"data_path: {args.data_path}")
    print(f"split_path: {args.split_path}")
    print(f"positive_count: {positive_count}")
    print(f"negative_count: {negative_count}")
    print(f"pos_weight: {pos_weight:.4f}")


if __name__ == "__main__":
    main()


"""
Example:
python CHAP2/pre_process/find_label_dist.py \
  --data-path /niddk-data-central/iWatch/pre_processed_pt/W \
  --split-path /niddk-data-central/iWatch/support_files/iwatch_split_dict.pkl \
  --bi-lstm-win-size 42 \
  --batch-size 32
"""
