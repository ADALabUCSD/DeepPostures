# Copyright 2024 Animesh Kumar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys

sys.path.append("./")

import random
import math
import argparse
import json
import numpy as np

from tqdm import tqdm
from commons import get_dataloaders
from model import CNNBiLSTMModel

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds
random.seed(2019)
np.random.seed(2019)


def mean_per_class_accuracy(predictions, labels, num_classes):
    per_class_accuracy = torch.zeros(num_classes)
    class_counts = torch.zeros(num_classes)

    for target, pred in zip(labels, predictions):
        per_class_accuracy[target] += torch.eq(target, pred).item()
        class_counts[target] += 1

    # Avoid division by zero
    class_counts[class_counts == 0] = 1

    mean_accuracy = torch.sum(per_class_accuracy / class_counts) / num_classes
    return mean_accuracy.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Argument parser for training CNN BiLSTM model."
    )
    optional_arguments = parser._action_groups.pop()
    required_arguments = parser.add_argument_group("required arguments")
    required_arguments.add_argument(
        "--pre-processed-dir",
        help="Pre-processed data directory",
        required=False,
    )

    optional_arguments.add_argument(
        "--transfer-learning-model",
        help="Transfer learning model name (default: CHAP_ALL_ADULTS)",
        default=None,
        required=False,
        choices=["CHAP_ALL_ADULTS", "CHAP_AUSDIAB", "NONE"],
    )
    optional_arguments.add_argument(
        "--learning-rate",
        help="Learning rate for training the model (default: 0.0001)",
        default=1e-4,
        type=float,
        required=False,
    )
    optional_arguments.add_argument(
        "--num-epochs",
        help="Number of epochs to train the model (default: 15)",
        default=15,
        type=int,
        required=False,
    )
    optional_arguments.add_argument(
        "--batch-size",
        help="Training batch size (default: 16)",
        default=16,
        type=int,
        required=False,
    )

    optional_arguments.add_argument(
        "--amp-factor",
        help="Factor to increase the number of neurons in the CNN layers (default: 2)",
        default=2,
        type=int,
        required=False,
    )
    optional_arguments.add_argument(
        "--cnn-window-size",
        help="CNN window size in seconds on which the predictions to be made (default: 10)",
        default=10,
        type=int,
        required=False,
    )
    optional_arguments.add_argument(
        "--bi-lstm-window-size",
        help="BiLSTM window size in minutes on which the predictions to be smoothed (default: 7)",
        default=7,
        type=int,
        required=False,
    )
    # No buffer based implementation is supported in PyTorch
    # optional_arguments.add_argument(
    #     "--shuffle-buffer-size",
    #     help="Training data shuffle buffer size in terms of number of records (default: 10000)",
    #     default=10000,
    #     type=int,
    #     required=False,
    # )
    optional_arguments.add_argument(
        "--training-data-fraction",
        help="Percentage of subjects to be used for training (default: 60)",
        default=60,
        type=int,
        required=False,
    )
    optional_arguments.add_argument(
        "--validation-data-fraction",
        help="Percentage of subjects to be used for validation (default: 20)",
        default=20,
        type=int,
        required=False,
    )
    optional_arguments.add_argument(
        "--testing-data-fraction",
        help="Percentage of subjects to be used for testing (default: 20)",
        default=20,
        type=int,
        required=False,
    )
    optional_arguments.add_argument(
        "--model-checkpoint-path",
        help="Path where the trained model will be saved (default: ./model-checkpoint)",
        default="./model-checkpoint",
        required=False,
    )

    optional_arguments.add_argument(
        "--num-classes",
        help="Number of classes in the training dataset (default: 2)",
        default=2,
        type=int,
        required=False,
    )
    optional_arguments.add_argument(
        "--class-weights",
        help="Class weights for loss aggregation (default: [1.0, 1.0])",
        required=False,
    )
    optional_arguments.add_argument(
        "--down-sample-frequency",
        help="Downsample frequency in Hz for GT3X data (default: 10)",
        default=10,
        type=int,
        required=False,
    )
    optional_arguments.add_argument(
        "--silent",
        help="Whether to hide info messages",
        default=False,
        required=False,
        action="store_true",
    )
    parser._action_groups.append(optional_arguments)
    args = parser.parse_args()

    if os.path.exists(args.model_checkpoint_path):
        raise Exception(
            "Model checkpoint: {} already exists.".format(args.model_checkpoint_path)
        )

    if args.transfer_learning_model:
        if args.transfer_learning_model == "CHAP_ALL_ADULTS":
            args.amp_factor = 2
            args.cnn_window_size = 10
            args.bi_lstm_window_size = 7
        elif args.transfer_learning_model == "CHAP_AUSDIAB":
            args.amp_factor = 4
            args.cnn_window_size = 10
            args.bi_lstm_window_size = 9
        elif args.transfer_learning_model != "NONE":
            raise Exception(
                "Unsupported transfer learning model: {}".format(
                    args.transfer_learning_model
                )
            )

    assert (
        args.training_data_fraction
        + args.validation_data_fraction
        + args.testing_data_fraction
    ) == 100, "Train, validation,test split fractions should add up to 100%"

    subject_ids = [fname.split(".")[0] for fname in os.listdir(args.pre_processed_dir)]
    random.shuffle(subject_ids)

    n_train_subjects = int(
        math.ceil(len(subject_ids) * args.training_data_fraction / 100.0)
    )
    train_subjects = subject_ids[:n_train_subjects]
    subject_ids = subject_ids[n_train_subjects:]

    if (100.0 - args.training_data_fraction) > 0:
        test_frac = (
            args.testing_data_fraction / (100.0 - args.training_data_fraction) * 100
        )
    else:
        test_frac = 0.0
    n_test_subjects = int(math.ceil(len(subject_ids) * test_frac / 100.0))
    test_subjects = subject_ids[:n_test_subjects]
    valid_subjects = subject_ids[n_test_subjects:]

    output_shapes = (
        (
            args.bi_lstm_window_size * (60 // args.cnn_window_size),
            args.cnn_window_size * args.down_sample_frequency,
            3,
        ),
        (args.bi_lstm_window_size * (60 // args.cnn_window_size)),
    )
    bi_lstm_win_size = 60 // args.down_sample_frequency * args.bi_lstm_window_size

    # Load class weights
    class_weights = None
    if args.class_weights:
        class_weights = json.loads(args.class_weights)
        class_weights = torch.tensor(class_weights)

    # Load model
    model = CNNBiLSTMModel(args.amp_factor, bi_lstm_win_size, args.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set optimizer and Loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if not args.silent:
        print("Training subjects: {}".format(train_subjects))
        print("Validation subjects: {}".format(valid_subjects))
        print("Testing subjects: {}".format(test_subjects))

    for epoch in tqdm(range(args.num_epochs)):
        train_dataloader, valid_dataloader, _ = get_dataloaders(
            pre_processed_dir=args.pre_processed_dir,
            bi_lstm_win_size=bi_lstm_win_size,
            batch_size=args.batch_size,
            train_subjects=train_subjects,
            valid_subjects=valid_subjects,
            test_subjects=None,
        )

        model.train()
        running_loss = 0.0
        training_accuracy = 0.0
        n_batches = 0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(
                device, dtype=torch.float32
            )

            inputs = inputs.view(
                -1, args.cnn_window_size * args.down_sample_frequency, 3, 1
            )
            inputs = inputs.permute(0, 3, 1, 2)
            labels = labels.view(-1, bi_lstm_win_size)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1, args.num_classes)
            labels = labels.view(-1)
            labels_one_hot = torch.nn.functional.one_hot(
                labels.long(), num_classes=args.num_classes
            )
            labels = labels_one_hot.view(-1, args.num_classes).to(torch.float32)

            # Calulate accuracy
            batch_acc = mean_per_class_accuracy(
                torch.argmax(outputs, dim=1),
                torch.argmax(labels, dim=1),
                args.num_classes,
            )

            training_accuracy += batch_acc

            # Calculate loss
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            n_batches += 1

        epoch_loss = running_loss / n_batches
        epoch_accuracy = training_accuracy / n_batches

        # Validation loop
        model.eval()
        val_loss = 0.0
        validation_accuracy = 0.0
        n_batches = 0
        if valid_dataloader:
            with torch.no_grad():
                for inputs, labels in valid_dataloader:
                    inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(
                        device, dtype=torch.float32
                    )

                    inputs = inputs.view(
                        -1, args.cnn_window_size * args.down_sample_frequency, 3, 1
                    )
                    inputs = inputs.permute(0, 3, 1, 2)
                    labels = labels.view(-1, bi_lstm_win_size)

                    outputs = model(inputs)
                    outputs = outputs.view(-1, args.n_classes)
                    labels = labels.view(-1)

                    labels_one_hot = torch.nn.functional.one_hot(
                        labels.long(), num_classes=args.num_classes
                    )
                    labels = labels_one_hot.view(-1, args.num_classes).to(torch.float32)

                    # Calulate accuracy
                    batch_acc = mean_per_class_accuracy(
                        torch.argmax(outputs, dim=1),
                        torch.argmax(labels, dim=1),
                        args.num_classes,
                    )
                    validation_accuracy += batch_acc

                    # Calculate loss
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    n_batches += 1

            epoch_val_loss = val_loss / n_batches
            epoch_val_acc = validation_accuracy / n_batches

            if not args.silent:
                print(
                    f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {training_accuracy:.2%}, Val Loss: {val_loss:.4f}, Val Accuracy: {epoch_val_acc:.2%}"
                )
        else:
            if not args.silent:
                print(
                    f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2%}"
                )
    if not args.silent:
        print("Training finished.")

        if not os.path.exists(args.model_checkpoint_path):
            os.makedirs(args.model_checkpoint_path)
        torch.save(
            model.state_dict(),
            os.path.join(args.model_checkpoint_path, "CUSTOM_MODEL.pth"),
        )
        print("Model saved in path: {}".format(args.model_checkpoint_path))
