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
import pandas as pd
import time

from tqdm import tqdm
from commons import get_dataloaders
from utils import write_metrics_to_csv, load_model_weights
from model import CNNBiLSTMModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds
random.seed(2019)
np.random.seed(2019)


def custom_transfer_learning_model_config(args):
    # Set these parameters for you custom model
    # args.amp_factor = 2
    # args.cnn_window_size = 10
    # args.bi_lstm_window_size = 7
    raise NotImplementedError("Define model config for your custom pre-trained model")


def create_splits(
    subject_ids,
    split_data_file,
    training_data_fraction,
    validation_data_fraction,
    testing_data_fraction,
    run_test,
):
    """
    Creates train/validation/tets split from split data file or based on data fractions
    """
    if split_data_file:
        # Read data from split data file
        df = pd.read_csv(split_data_file)
        train_subjects = (
            df[df["type"].str.contains("train", na=False)]["study_id"]
            .astype(str)
            .to_list()
        )
        print("P2 random Train subjects: ", len(train_subjects))
        # remove missing elements
        missing_train_subjects = set(train_subjects) - set(subject_ids)
        train_subjects = list(set(train_subjects) - set(missing_train_subjects))
        # Retain state of randomization later
        train_subjects.sort()
        random.shuffle(train_subjects)

        valid_subjects = []
        if validation_data_fraction:
            print("Validation data fraction: ", validation_data_fraction)
            train_subjects, valid_subjects = train_test_split(
                train_subjects,
                test_size=validation_data_fraction / 100.0,
                shuffle=False,
            )
        if training_data_fraction < 100:
            print("Training data fraction: ", training_data_fraction)
            train_subjects, _ = train_test_split(
                train_subjects, train_size=training_data_fraction / 100.0, shuffle=False
            )

        test_subjects = []
        missing_test_subjects = []
        if run_test:
            test_subjects = (
                df[df["type"].str.contains("test", na=False)]["study_id"]
                .astype(str)
                .to_list()
            )
            print("P2 random Test subjects: ", len(test_subjects))
            missing_test_subjects = set(test_subjects) - set(subject_ids)
            test_subjects = list(set(test_subjects) - set(missing_test_subjects))
        # Retain state of randomization later
        test_subjects.sort()
        if missing_train_subjects or missing_test_subjects:
            print(
                f"""Following subjects are missing from preprocessed directory and will be skipped:
                  Train subject: {missing_train_subjects}
                  Test Subjects: {missing_test_subjects}"""
            )
        print(
            "Splits created: Train subjects: ",
            len(train_subjects),
            "Valid subjects: ",
            len(valid_subjects),
            "Test subjects: ",
            len(test_subjects),
        )

        return train_subjects, valid_subjects, test_subjects
    else:
        assert (
            args.training_data_fraction
            + args.validation_data_fraction
            + args.testing_data_fraction
        ) == 100, "Train, validation,test split fractions should add up to 100%"

        random.shuffle(subject_ids)
        n_train_subjects = int(
            math.ceil(len(subject_ids) * training_data_fraction / 100.0)
        )
        train_subjects = subject_ids[:n_train_subjects]
        subject_ids = subject_ids[n_train_subjects:]

        if (100.0 - training_data_fraction) > 0:
            test_frac = testing_data_fraction / (100.0 - training_data_fraction) * 100
        else:
            test_frac = 0.0
        n_test_subjects = int(math.ceil(len(subject_ids) * test_frac / 100.0))
        test_subjects = subject_ids[:n_test_subjects]
        valid_subjects = subject_ids[n_test_subjects:]

        return train_subjects, valid_subjects, test_subjects


if __name__ == "__main__":
    main_start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Argument parser for training CNN BiLSTM model."
    )
    optional_arguments = parser._action_groups.pop()
    required_arguments = parser.add_argument_group("required arguments")
    required_arguments.add_argument(
        "--pre-processed-dir",
        help="Pre-processed data directory",
        required=True,
    )

    optional_arguments.add_argument(
        "--transfer-learning-model",
        help="Transfer learning model name (default: CHAP_ALL_ADULTS)",
        default=None,
        required=False,
        choices=["CHAP_ALL_ADULTS", "CHAP_AUSDIAB", "CUSTOM_MODEL"],
    )
    optional_arguments.add_argument(
        "--learning-rate",
        help="Learning rate for training the model (default: 0.0001)",
        default=1e-4,
        type=float,
        required=False,
    )
    optional_arguments.add_argument(
        "--weight-decay",
        help="L2 regulatization weight decay",
        type=float,
        default=0.0,
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
    optional_arguments.add_argument(
        "--output-file",
        help="Output file to log training metric",
        default="./output_metrics.csv",
        required=False,
    )
    optional_arguments.add_argument(
        "--split-data-file",
        help="CSV file containing train//test split subject id in separate columns",
        required=False,
    )
    optional_arguments.add_argument(
        "--run-test",
        default=False,
        required=False,
        action="store_true",
    )
    optional_arguments.add_argument(
        "--run-sanity-validation",
        default=False,
        required=False,
        action="store_true",
    )
    optional_arguments.add_argument(
        "--model-checkpoint-interval",
        default=1,
        required=False,
        type=int,
    )
    optional_arguments.add_argument(
        "--lr-scheduler",
        default=None,
        required=False,
        choices=["linear"],
    )

    parser._action_groups.append(optional_arguments)
    args = parser.parse_args()

    print("Using device", "cuda" if torch.cuda.is_available() else "cpu")
    print("Arguments: ", args)

    # Precheck on directories
    if os.path.exists(args.model_checkpoint_path):
        raise Exception(
            "Model checkpoint: {} already exists.".format(args.model_checkpoint_path)
        )
    if not os.path.exists(os.path.join(args.model_checkpoint_path, "checkpoint")):
        os.makedirs(os.path.join(args.model_checkpoint_path, "checkpoint"))

    if args.transfer_learning_model:
        if args.transfer_learning_model == "CUSTOM_MODEL":
            custom_transfer_learning_model_config()
            transfer_learning_model_path = os.path.join(
                "./model-checkpoint", f"{args.transfer_learning_model}.pth"
            )
        if args.transfer_learning_model == "CHAP_ALL_ADULTS":
            args.amp_factor = 2
            args.cnn_window_size = 10
            args.bi_lstm_window_size = 7
            transfer_learning_model_path = os.path.join(
                "./pre-trained-models-pt", f"{args.transfer_learning_model}.pth"
            )
        elif args.transfer_learning_model == "CHAP_AUSDIAB":
            args.amp_factor = 4
            args.cnn_window_size = 10
            args.bi_lstm_window_size = 9
            transfer_learning_model_path = os.path.join(
                "./pre-trained-models-pt", f"{args.transfer_learning_model}.pth"
            )
        else:
            raise Exception(
                "Unsupported transfer learning model: {}".format(
                    args.transfer_learning_model
                )
            )

    subject_ids = [
        fname.split(".")[0][:-2]
        for fname in os.listdir(os.path.join(args.pre_processed_dir, "BL"))
    ]
    subject_ids += [
        fname.split(".")[0][:-2]
        for fname in os.listdir(os.path.join(args.pre_processed_dir, "FV"))
    ]
    subject_ids = list(set(subject_ids))

    print("Subject IDs: ", subject_ids)
    train_subjects, valid_subjects, test_subjects = create_splits(
        subject_ids,
        args.split_data_file,
        args.training_data_fraction,
        args.validation_data_fraction,
        args.testing_data_fraction,
        args.run_test,
    )

    train_subjects = [x + "BL" for x in train_subjects] + [
        x + "FV" for x in train_subjects
    ]
    valid_subjects = [x + "BL" for x in valid_subjects] + [
        x + "FV" for x in valid_subjects
    ]
    test_subjects = [x + "BL" for x in test_subjects] + [
        x + "FV" for x in test_subjects
    ]
    random.shuffle(train_subjects)
    random.shuffle(valid_subjects)
    random.shuffle(test_subjects)

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

    if args.transfer_learning_model:
        load_model_weights(model, transfer_learning_model_path, weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not args.silent:
        print("Training on {} subjects: {}".format(len(train_subjects), train_subjects))
        print(
            "Validation on {} subjects: {}".format(len(valid_subjects), valid_subjects)
        )
        print("Testing on {} subjects: {}".format(len(test_subjects), test_subjects))

    # Set optimizer and Loss function
    criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler:
        if args.lr_scheduler == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.num_epochs)
    metrics = []

    # Load dataloaders
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(
        pre_processed_dir=args.pre_processed_dir,
        bi_lstm_win_size=bi_lstm_win_size,
        batch_size=args.batch_size,
        train_subjects=train_subjects,
        valid_subjects=valid_subjects,
        test_subjects=test_subjects if test_subjects else None,
    )

    if args.run_sanity_validation:
        print("Running sanity validation")
        # Validation loop
        model.eval()
        validation_accuracy = 0.0
        n_batches = 0
        epoch_val_acc = None
        if valid_dataloader != None:
            with torch.no_grad():
                for inputs, labels in valid_dataloader:
                    inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(
                        device, dtype=torch.float32
                    )

                    inputs = inputs.view(
                        -1, args.cnn_window_size * args.down_sample_frequency, 3, 1
                    )
                    # convert to (N, H, W, C) to (N, C, H, W)
                    inputs = inputs.permute(0, 3, 1, 2)
                    labels = labels.view(-1, bi_lstm_win_size)
                    # outputs
                    outputs = model(inputs)
                    # convert to 1D tensor
                    outputs = outputs.view(-1)
                    labels = labels.view(-1)
                    # convert label to one hot
                    labels_one_hot = torch.nn.functional.one_hot(
                        labels.long(), num_classes=args.num_classes
                    )
                    labels = labels_one_hot.view(-1, args.num_classes)
                    labels = torch.argmax(labels, dim=1).to(torch.float32)
                    # Calulate accuracy
                    preds = torch.round(torch.sigmoid(outputs))
                    batch_acc = accuracy_score(
                        preds.cpu().detach().numpy(), labels.cpu().detach().numpy()
                    )
                    validation_accuracy += batch_acc

                    # Calculate loss
                    loss = criterion(outputs, labels)
                    n_batches += 1

            epoch_val_acc = validation_accuracy / n_batches
        print(f"Sanity Validation Accuracy: {epoch_val_acc:.2%}")
    print("Running Training")
    for epoch in tqdm(range(args.num_epochs)):
        start_time = time.time()  # Start the timer for the epoch
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
            # convert to (N, H, W, C) to (N, C, H, W)
            inputs = inputs.permute(0, 3, 1, 2)
            labels = labels.view(-1, bi_lstm_win_size)

            optimizer.zero_grad()
            # outputs
            outputs = model(inputs)
            # convert to 1D tensor
            outputs = outputs.view(-1)
            labels = labels.view(-1)
            # convert label to one hot
            labels_one_hot = torch.nn.functional.one_hot(
                labels.long(), num_classes=args.num_classes
            )
            labels = labels_one_hot.view(-1, args.num_classes)
            labels = torch.argmax(labels, dim=1).to(torch.float32)
            # Calulate accuracy
            preds = torch.round(torch.sigmoid(outputs))
            batch_acc = accuracy_score(
                preds.cpu().detach().numpy(), labels.cpu().detach().numpy()
            )

            training_accuracy += batch_acc
            # Calculate loss
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            n_batches += 1

        epoch_train_loss = running_loss / n_batches
        epoch_train_accuracy = training_accuracy / n_batches

        # Validation loop
        model.eval()
        val_loss = 0.0
        validation_accuracy = 0.0
        n_batches = 0
        epoch_val_loss = None
        epoch_val_acc = None
        if valid_dataloader != None:
            with torch.no_grad():
                for inputs, labels in valid_dataloader:
                    inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(
                        device, dtype=torch.float32
                    )

                    inputs = inputs.view(
                        -1, args.cnn_window_size * args.down_sample_frequency, 3, 1
                    )
                    # convert to (N, H, W, C) to (N, C, H, W)
                    inputs = inputs.permute(0, 3, 1, 2)
                    labels = labels.view(-1, bi_lstm_win_size)
                    # outputs
                    outputs = model(inputs)
                    # convert to 1D tensor
                    outputs = outputs.view(-1)
                    labels = labels.view(-1)
                    # convert label to one hot
                    labels_one_hot = torch.nn.functional.one_hot(
                        labels.long(), num_classes=args.num_classes
                    )
                    labels = labels_one_hot.view(-1, args.num_classes)
                    labels = torch.argmax(labels, dim=1).to(torch.float32)
                    # Calulate accuracy
                    preds = torch.round(torch.sigmoid(outputs))
                    batch_acc = accuracy_score(
                        preds.cpu().detach().numpy(), labels.cpu().detach().numpy()
                    )
                    validation_accuracy += batch_acc

                    # Calculate loss
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    n_batches += 1

            epoch_val_loss = val_loss / n_batches
            epoch_val_acc = validation_accuracy / n_batches

        end_time = time.time()
        epoch_duration = end_time - start_time
        if valid_dataloader != None:
            if not args.silent:
                print(
                    f"Epoch [{epoch+1}/{args.num_epochs}], Runtime: {epoch_duration:.2f} seconds, Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2%}, Val Loss: {val_loss:.4f}, Val Accuracy: {epoch_val_acc:.2%}"
                )
        else:
            if not args.silent:
                print(
                    f"Epoch [{epoch+1}/{args.num_epochs}], Runtime: {epoch_duration:.2f} seconds, Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2%}"
                )
        # Add a new entry for the current epoch
        metrics.append(
            {
                "epoch": epoch + 1,
                "runtime": epoch_duration,
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_accuracy,
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc,
            }
        )
        # Save model checkpoint
        if (
            args.model_checkpoint_interval
            and epoch % args.model_checkpoint_interval == 0
        ):
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    # ... other items you want to save
                },
                os.path.join(
                    os.path.join(args.model_checkpoint_path, "checkpoint"),
                    f"checkpoint_epoch_{epoch}.pth",
                ),
            )
        # Step the scheduler
        if scheduler:
            scheduler.step()
            print(f"Learning rate: {scheduler.get_last_lr()}")

    # Log metric values
    write_metrics_to_csv(metrics, args.output_file)
    # Save model
    if not args.silent:
        print("Training finished.")

    if not os.path.exists(args.model_checkpoint_path):
        os.makedirs(args.model_checkpoint_path)
    torch.save(
        model.state_dict(),
        os.path.join(args.model_checkpoint_path, "CUSTOM_MODEL.pth"),
    )
    print("Model saved in path: {}".format(args.model_checkpoint_path))

    # Testing pipeline
    if test_subjects:
        print("Running Testing")
        del model
        model = CNNBiLSTMModel(args.amp_factor, bi_lstm_win_size, args.num_classes)
        load_model_weights(
            model,
            os.path.join(args.model_checkpoint_path, "CUSTOM_MODEL.pth"),
            weights_only=True,
        )
        model.to(device)
        model.eval()

        test_accuracy = 0.0
        n_batches = 0
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(
                    device, dtype=torch.float32
                )

                inputs = inputs.view(
                    -1, args.cnn_window_size * args.down_sample_frequency, 3, 1
                )
                # convert to (N, H, W, C) to (N, C, H, W)
                inputs = inputs.permute(0, 3, 1, 2)
                labels = labels.view(-1, bi_lstm_win_size)
                # outputs
                outputs = model(inputs)
                # convert to 1D tensor
                outputs = outputs.view(-1)
                labels = labels.view(-1)
                # convert label to one hot
                labels_one_hot = torch.nn.functional.one_hot(
                    labels.long(), num_classes=args.num_classes
                )
                labels = labels_one_hot.view(-1, args.num_classes)
                labels = torch.argmax(labels, dim=1).to(torch.float32)
                # Calulate accuracy
                preds = torch.round(torch.sigmoid(outputs))
                batch_acc = accuracy_score(
                    preds.cpu().detach().numpy(), labels.cpu().detach().numpy()
                )
                test_accuracy += batch_acc

                n_batches += 1

            test_accuracy = test_accuracy / n_batches
        print(f"Test Accuracy: {test_accuracy:.2%}")
    main_end_time = time.time()
    print(f"Done!!\nTotal time taken: {main_end_time - main_start_time:.2f} seconds")