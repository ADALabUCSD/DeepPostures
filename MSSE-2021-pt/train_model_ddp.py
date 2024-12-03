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
from commons import get_dataloaders_dist
from utils import (
    write_metrics_to_csv,
    load_model_weights,
    compute_accuracy_from_confusion_matrix,
)
from model import CNNBiLSTMModel
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# Set random seeds
random.seed(2019)
np.random.seed(2019)


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"  # The address of the master node
    os.environ["MASTER_PORT"] = "12355"  # A free port for communication

    dist.init_process_group(
        backend="nccl",  # Backend, 'nccl' works well with GPUs
        rank=rank,
        world_size=world_size,
    )


def cleanup_ddp():
    dist.destroy_process_group()


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


def train(
    rank,
    world_size,
    args,
    bi_lstm_win_size,
    class_weights,
    transfer_learning_model_path,
    train_subjects,
    valid_subjects,
    test_subjects,
):
    # setup
    setup_ddp(rank, world_size)

    print(f"Setup done for rank {rank}")
    device = torch.device(f"cuda:{rank}")

    # Load model
    model = CNNBiLSTMModel(args.amp_factor, bi_lstm_win_size, args.num_classes).to(
        device
    )

    if args.transfer_learning_model:
        load_model_weights(model, transfer_learning_model_path, weights_only=False)

    model = DDP(model, device_ids=[rank])

    # Set optimizer and Loss function
    criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = None
    if args.lr_scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.num_epochs
        )
    metrics = []

    # Load dataloaders
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders_dist(
        pre_processed_dir=args.pre_processed_dir,
        bi_lstm_win_size=bi_lstm_win_size,
        batch_size=args.batch_size,
        train_subjects=train_subjects,
        valid_subjects=valid_subjects,
        test_subjects=test_subjects if test_subjects else None,
        rank=rank,
        world_size=world_size,
    )

    if args.run_sanity_validation:
        print(f"Running sanity validation in rank - {rank}")

        # Validation loop
        model.eval()
        # Initialize confusion matrix for the current epoch
        cm_sanity_val = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)

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

                    # Compute confusion matrix for the batch
                    batch_cm = confusion_matrix(
                        labels.cpu().numpy(),
                        preds.cpu().numpy(),
                        labels=np.arange(args.num_classes),
                    )
                    cm_sanity_val += batch_cm

            # Reduce confusion matrix across all processes
            cm_sanity_val_tensor = torch.tensor(cm_sanity_val, device=rank)
            dist.reduce(
                cm_sanity_val_tensor, dst=0, op=dist.ReduceOp.SUM
            )  # Aggregate on rank 0

            if rank == 0:
                # Compute accuracy on rank 0
                cm_sanity_val = cm_sanity_val_tensor.cpu().numpy()
                sanity_val_accuracy, sanity_val_balanced_accuracy = (
                    compute_accuracy_from_confusion_matrix(cm_sanity_val)
                )
                print(
                    f"Sanity Validation Accuracy: {sanity_val_accuracy:.2%} Balanced Accuracy: {sanity_val_balanced_accuracy:.2%}"
                )

    print("Running Training in rank - ", rank)
    for epoch in tqdm(range(args.num_epochs)):
        start_time = time.time()  # Start the timer for the epoch
        model.train()

        training_loss = 0.0
        n_batches_train = 0
        cm_train = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(
                device, dtype=torch.float32
            )
            batch_size = labels.shape[0]
            inputs = inputs.view(
                -1, args.cnn_window_size * args.down_sample_frequency, 3, 1
            )
            # convert to (N, H, W, C) to (N, C, H, W)
            inputs = inputs.permute(0, 3, 1, 2)

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
            # Compute confusion matrix for the batch
            batch_cm = confusion_matrix(
                labels.cpu().detach().numpy(),
                preds.cpu().detach().numpy(),
                labels=np.arange(args.num_classes),
            )
            cm_train += batch_cm

            # Convert back to batch size for faster calculation
            # Also BCE takes mean per batch
            outputs = outputs.view(batch_size, -1)
            labels = labels.view(batch_size, -1)

            # Calculate loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * outputs.size(0)
            n_batches_train += 1

        # Reduce confusion matrix across all processes
        cm_train_tensor = torch.tensor(cm_train, device=rank)
        training_loss_tensor = torch.tensor(training_loss, device=rank)
        n_batches_train_tensor = torch.tensor(n_batches_train, device=rank)

        dist.reduce(cm_train_tensor, dst=0, op=dist.ReduceOp.SUM)  # Aggregate on rank 0
        dist.reduce(
            training_loss_tensor, dst=0, op=dist.ReduceOp.SUM
        )  # Aggregate on rank 0
        dist.reduce(
            n_batches_train_tensor, dst=0, op=dist.ReduceOp.SUM
        )  # Aggregate on rank 0

        # Validation loop
        model.eval()
        val_loss = 0.0
        n_batches_val = 0
        cm_val = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
        if valid_dataloader != None:
            with torch.no_grad():
                for inputs, labels in valid_dataloader:
                    inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(
                        device, dtype=torch.float32
                    )
                    batch_size = labels.shape[0]

                    inputs = inputs.view(
                        -1, args.cnn_window_size * args.down_sample_frequency, 3, 1
                    )
                    # convert to (N, H, W, C) to (N, C, H, W)
                    inputs = inputs.permute(0, 3, 1, 2)
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
                    # Compute confusion matrix for the batch
                    batch_cm = confusion_matrix(
                        labels.cpu().detach().numpy(),
                        preds.cpu().detach().numpy(),
                        labels=np.arange(args.num_classes),
                    )
                    cm_val += batch_cm

                    # Convert back to batch size for faster calculation
                    # Also BCE takes mean per batch
                    outputs = outputs.view(batch_size, -1)
                    labels = labels.view(batch_size, -1)

                    # Calculate loss
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * outputs.size(0)
                    n_batches_val += 1

            # Reduce confusion matrix across all processes
            cm_val_tensor = torch.tensor(cm_val, device=rank)
            val_loss_tensor = torch.tensor(val_loss, device=rank)
            n_batches_val_tensor = torch.tensor(n_batches_val, device=rank)

            dist.reduce(
                cm_val_tensor, dst=0, op=dist.ReduceOp.SUM
            )  # Aggregate on rank 0
            dist.reduce(
                val_loss_tensor, dst=0, op=dist.ReduceOp.SUM
            )  # Aggregate on rank 0
            dist.reduce(
                n_batches_val_tensor, dst=0, op=dist.ReduceOp.SUM
            )  # Aggregate on rank 0

        end_time = time.time()
        epoch_duration = end_time - start_time

        # Metrics and checkpointing for the epoch
        if rank == 0:
            cm_train = cm_train_tensor.cpu().numpy()
            training_loss = training_loss_tensor.cpu().numpy()
            n_batches_train = n_batches_train_tensor.cpu().numpy()
            epoch_train_accuracy, epoch_train_balanced_accuracy = (
                compute_accuracy_from_confusion_matrix(cm_train)
            )
            epoch_train_loss = training_loss / n_batches_train

            if valid_dataloader != None:
                cm_val = cm_val_tensor.cpu().numpy()
                val_loss = val_loss_tensor.cpu().numpy()
                n_batches_val = n_batches_val_tensor.cpu().numpy()
                epoch_val_accuracy, epoch_val_balanced_accuracy = (
                    compute_accuracy_from_confusion_matrix(cm_val)
                )
                epoch_val_loss = val_loss / n_batches_val

                if not args.silent:
                    print(
                        f"Epoch [{epoch+1}/{args.num_epochs}], Runtime: {epoch_duration:.2f} seconds, Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2%}, Train Balanced Accuracy: {epoch_train_balanced_accuracy:.2%}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.2%}, Balanced Accuracy: {epoch_val_balanced_accuracy:.2%}"
                    )
            else:
                if not args.silent:
                    print(
                        f"Epoch [{epoch+1}/{args.num_epochs}], Runtime: {epoch_duration:.2f} seconds, Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2%}, Train Accuracy: {epoch_train_accuracy:.2%}, Train Balanced Accuracy: {epoch_train_balanced_accuracy:.2%}"
                    )
            # Add a new entry for the current epoch
            metrics.append(
                {
                    "epoch": epoch + 1,
                    "runtime": epoch_duration,
                    "train_loss": epoch_train_loss,
                    "train_acc": epoch_train_accuracy,
                    "train_balanced_acc": epoch_train_balanced_accuracy,
                    "val_loss": epoch_val_loss,
                    "val_acc": epoch_val_accuracy,
                    "val_balanced_acc": epoch_val_balanced_accuracy,
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
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        # ... other items you want to save
                    },
                    os.path.join(
                        os.path.join(args.model_checkpoint_path, "checkpoint"),
                        f"checkpoint_epoch_{epoch}.pth",
                    ),
                )
            # Log metric values
            write_metrics_to_csv(metrics, args.output_file)
            # Save model
            if not args.silent:
                print("Training finished.")

            # if the model is wrapped in DDP, unwrap it before saving
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                # Extract the underlying model
                model = model.module

            if not os.path.exists(args.model_checkpoint_path):
                os.makedirs(args.model_checkpoint_path)
            torch.save(
                model.state_dict(),
                os.path.join(args.model_checkpoint_path, "CUSTOM_MODEL.pth"),
            )
            print("Model saved in path: {}".format(args.model_checkpoint_path))

        # Step the scheduler
        if scheduler:
            scheduler.step()
            print(f"Learning rate: {scheduler.get_last_lr()}")

    # Testing pipeline
    if test_subjects:
        print("Running Testing in rank - ", rank)
        del model
        model = CNNBiLSTMModel(args.amp_factor, bi_lstm_win_size, args.num_classes).to(device)
        load_model_weights(
            model,
            os.path.join(args.model_checkpoint_path, "CUSTOM_MODEL.pth"),
            weights_only=True,
        )

        model = DDP(model, device_ids=[rank])
        model.eval()

        # Initialize confusion matrix for the current epoch
        cm_test = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
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

                # Compute confusion matrix for the batch
                batch_cm = confusion_matrix(
                    labels.cpu().numpy(),
                    preds.cpu().numpy(),
                    labels=np.arange(args.num_classes),
                )
                cm_test += batch_cm

            # Reduce confusion matrix across all processes
            cm_test_tensor = torch.tensor(cm_test, device=rank)
            dist.reduce(
                cm_test_tensor, dst=0, op=dist.ReduceOp.SUM
            )  # Aggregate on rank 0

            if rank == 0:
                # Compute accuracy on rank 0
                cm_test = cm_test_tensor.cpu().numpy()
                test_accuracy, test_balanced_accuracy = (
                    compute_accuracy_from_confusion_matrix(cm_test)
                )
                print(
                    f"Test Accuracy: {test_accuracy:.2%} Balanced Test Accuracy: {test_balanced_accuracy:.2%}"
                )

    # Clean up DDP
    cleanup_ddp()


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
        required=False,
        default=0.0,
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
        type=int,
        required=False,
    )
    optional_arguments.add_argument(
        "--lr-scheduler",
        default=None,
        required=False,
        choices=["linear"],
    )

    parser._action_groups.append(optional_arguments)
    args = parser.parse_args()

    print(
        "Using device",
        (
            f"cuda devices: {torch.cuda.device_count()}"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
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

    if not args.silent:
        print("Training on {} subjects: {}".format(len(train_subjects), train_subjects))
        print(
            "Validation on {} subjects: {}".format(len(valid_subjects), valid_subjects)
        )
        print("Testing on {} subjects: {}".format(len(test_subjects), test_subjects))

    # DDP setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    world_size = torch.cuda.device_count()

    mp.spawn(
        train,
        args=(
            world_size,
            args,
            bi_lstm_win_size,
            class_weights,
            transfer_learning_model_path,
            train_subjects,
            valid_subjects,
            test_subjects,
        ),
        nprocs=world_size,
    )

    main_end_time = time.time()
    print(f"Done!!\nTotal time taken: {main_end_time - main_start_time:.2f} seconds")
