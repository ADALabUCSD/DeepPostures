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
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from datetime import datetime, timedelta
from tqdm import tqdm
from functools import partial

# Custom IterableDataset for distributed training which divides the data among the workers
class IterDatasetDist(torch.utils.data.IterableDataset):
    """
    PyTorch IterableDataset created from a generator
    """

    def __init__(
        self,
        generator,
        rank,
        world_size,
        data_root=None,
        win_size_10s=None,
        subject_ids=None,
    ):
        self.generator = generator
        self.data_root = data_root
        self.win_size_10s = win_size_10s
        self.subject_ids = subject_ids
        self.rank = rank
        self.world_size = world_size

    # TODO: This does IO on the file multiple times.. should we divide files to different workers?
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            worker_id = 0
            num_workers = 1
        else:
            # In multi-worker mode
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Calculate the total number of chunks
        total_chunks = self.world_size * num_workers
        chunk_id = self.rank * num_workers + worker_id

        # Create an iterator for the assigned chunk
        iterator = iter(
            self.generator(self.data_root, self.win_size_10s, self.subject_ids)
        )
        for idx, item in enumerate(iterator):
            if idx % total_chunks == chunk_id:
                yield item


class IterDataset(torch.utils.data.IterableDataset):
    """
    PyTorch IterableDataset created from a generator
    """

    def __init__(self, generator, data_root=None, win_size_10s=None, subject_ids=None):
        self.generator = generator
        self.data_root = data_root
        self.win_size_10s = win_size_10s
        self.subject_ids = subject_ids

    def __iter__(self):
        return iter(self.generator(self.data_root, self.win_size_10s, self.subject_ids))

class IterDatasetSubject(torch.utils.data.IterableDataset):
    """
    PyTorch IterableDataset created from a generator
    """

    def __init__(self, generator, subject_data=[]):
        self.generator = generator
        self.subject_data = subject_data

    def __iter__(self):
        return iter(self.generator(self.subject_data))


def input_iterator(data_root, subject_id, train=False):
    """
    Iterate and read the preprocessed data files
    """

    fnames = [
        name.split(".")[0]
        for name in os.listdir(os.path.join(data_root, subject_id))
        if not name.startswith(".")
    ]
    fnames.sort()
    for i in range(len(fnames) - 1):
        try:
            assert datetime.strptime(fnames[i + 1], "%Y-%m-%d").date() - datetime.strptime(
                fnames[i], "%Y-%m-%d"
            ).date() == timedelta(days=1)
        except AssertionError:
            message = f"subject_id: {subject_id}, non-consecutive dates: {fnames[i]} -> {fnames[i + 1]}\n"
            print(message.strip())
            with open("error_log.txt", "a") as f:
                f.write(message)

    data_batch = []
    timestamps_batch = []
    label_batch = []
    for fname in fnames:
        h5f = h5py.File(os.path.join(data_root, subject_id, "{}.h5".format(fname)), "r")
        timestamps = h5f.get("time")[:]
        data = h5f.get("data")[:]
        sleeping = h5f.get("sleeping")[:]
        non_wear = h5f.get("non_wear")[:]
        label = h5f.get("label")[:]

        for d, t, s, nw, l in zip(data, timestamps, sleeping, non_wear, label):
            # if train and l == -1:
            #     raise Exception('Missing ground truth label information in pre-processed data')

            if s == 1 or nw == 1 or (train and l == -1):
                if len(timestamps_batch) > 0:
                    yield np.array(data_batch), np.array(timestamps_batch), np.array(
                        label_batch
                    )
                data_batch = []
                timestamps_batch = []
                label_batch = []
                continue

            data_batch.append(d)
            timestamps_batch.append(t)
            label_batch.append(l)

        h5f.close()

    if len(timestamps_batch) > 0:
        yield np.array(data_batch), np.array(timestamps_batch), np.array(label_batch)


def window_generator(data_root, win_size_10s, subject_ids,transform=None):
    """
    Generate windowed to be processed by CNN
    """

    for subject_id in tqdm(subject_ids):
        subject_dir = os.path.join(data_root, subject_id)
        if os.path.isdir(subject_dir):
            for x_seq, _, y_seq in input_iterator(
                data_root, subject_id, train=True
            ):
                x_window = []
                y_window = []
                for x, y in zip(x_seq, y_seq):
                    x_window.append(x)
                    y_window.append(y)

                    if len(y_window) == win_size_10s:
                        x_sample = np.stack(x_window, axis=0) # num_window, 100, 3
                        y_sample = np.stack(y_window, axis=0)

                        if transform is not None:
                            x_sample = transform(x_sample)

                        yield x_sample, y_sample

                        x_window = []
                        y_window = []
        else:
            print("Subject data at {} not found".format(subject_dir))


def get_subject_dataloader(test_subjects_data, batch_size):
    """
    Get dataloader for a single subject from preprocessed data
    """

    def list_generator(lst):
        for item in lst:
            yield item

    subject_data = IterDatasetSubject(list_generator, test_subjects_data)
    subject_dataloader = DataLoader(
        subject_data, batch_size=batch_size, pin_memory=True
    )
    return subject_dataloader


def get_dataloaders_dist(
    pre_processed_dir,
    bi_lstm_win_size,
    batch_size,
    train_subjects,
    valid_subjects,
    test_subjects,
    rank,
    world_size,
    transform,
):
    """
    Process data and get dataloaders for subject
    """

    train_dataloader = None
    valid_dataloader = None
    test_dataloader = None
    # https://github.com/pytorch/ignite/issues/1076#issuecomment-829191233
    if train_subjects:
        gen_with_transform = partial(window_generator, transform=transform)

        train_data = IterDatasetDist(
            gen_with_transform,
            rank,
            world_size,
            pre_processed_dir,
            bi_lstm_win_size,
            train_subjects,
        )
        
        shuffled_train_data = BufferedShuffleDataset(train_data, buffer_size=2*batch_size)
        train_dataloader = DataLoader(
            shuffled_train_data, 
            batch_size=batch_size, 
            pin_memory=True,
            num_workers=4, 
            worker_init_fn=worker_init_fn) 

    if valid_subjects:
        # validation data does not need to be shuffled
        valid_data = IterDatasetDist(
            window_generator,
            rank,
            world_size,
            pre_processed_dir,
            bi_lstm_win_size,
            valid_subjects,
        )
        valid_dataloader = DataLoader(
            valid_data, 
            batch_size=batch_size, 
            pin_memory=True,
            num_workers=2, 
        )
        
    if test_subjects:
        test_data = IterDatasetDist(
            window_generator,
            rank,
            world_size,
            pre_processed_dir,
            bi_lstm_win_size,
            test_subjects,
        )
        test_dataloader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

    return train_dataloader, valid_dataloader, test_dataloader


def get_dataloaders(
    pre_processed_dir,
    bi_lstm_win_size,
    batch_size,
    train_subjects,
    valid_subjects,
    test_subjects,
):
    """
    Process data and get dataloaders for subject
    """

    train_dataloader = None
    valid_dataloader = None
    test_dataloader = None

    if train_subjects:
        train_data = IterDataset(
            window_generator, pre_processed_dir, bi_lstm_win_size, train_subjects
        )
        train_dataloader = DataLoader(
            train_data, batch_size=batch_size, pin_memory=True,
            drop_last=True,  # Drop the last incomplete batch
        )

    if valid_subjects:
        valid_data = IterDataset(
            window_generator, pre_processed_dir, bi_lstm_win_size, valid_subjects,
        )
        valid_dataloader = DataLoader(
            valid_data, batch_size=batch_size, pin_memory=True,
            drop_last=False,
        )
    if test_subjects:
        test_data = IterDataset(
            window_generator, pre_processed_dir, bi_lstm_win_size, test_subjects
        )
        test_dataloader = DataLoader(test_data, batch_size=batch_size, pin_memory=True,drop_last=False,)

    return train_dataloader, valid_dataloader, test_dataloader


# Copyright 2025 Leo Chen
# All rights reserved.
from torch.utils.data import IterableDataset
import random

def worker_init_fn(worker_id):
    # usage: worker_init_fn is used to set the random seed for each worker
    # This ensures that each worker has a different random seed
    # and can produce different random numbers
    # This is important for data augmentation and shuffling

    seed = torch.initial_seed() % 2**32
    random.seed(seed)

class BufferedShuffleDataset(IterableDataset):
    '''
    Useage: wrapper class for shuffling data in an iterable dataset
    Tips: buffer_size dependes on CPU memory, try to start with 2*batch_size
    '''
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = []
        iterator = iter(self.dataset)
        try:
            for _ in range(self.buffer_size):
                buffer.append(next(iterator))
        except StopIteration:
            pass  # In case the dataset has fewer items than the buffer size

        while buffer:
            try:
                item = next(iterator)
                idx = random.randint(0, len(buffer) - 1)
                yield buffer[idx]
                buffer[idx] = item
            except StopIteration:
                break

        while buffer:
            yield buffer.pop(random.randint(0, len(buffer) - 1))

# def data_aug(x):
#     '''
#     Inout: x: numpy array of shape (num_windows, 100, 3)
#     Output: x: numpy array of shape (num_windows, 100, 3) with data augmentation applied

#     https://shamilmamedov.com/blog/2023/da-time-series/
#     '''

#     num_windows, T, C = x.shape
#     x_aug = x.copy()

#     # Channel swapping: apply one permutation across all windows
#     perm = np.random.permutation(C)
#     x_aug = x_aug[:, :, perm]

#     # Jittering: add Gaussian noise
#     noise_std = 0.05  # Adjust as needed
#     noise = np.random.normal(loc=0.0, scale=noise_std, size=x_aug.shape)
#     x_aug += noise

#     # Scaling: apply one random scalar per window
#     scaling_factors = np.random.normal(loc=1.0, scale=0.1, size=(num_windows, 1, 1))
#     x_aug *= scaling_factors

#     return x_aug

from torch.utils.data import DataLoader, Subset
from collections import defaultdict
def get_subjectwise_dataloaders(dataset, batch_size=32, num_workers=4, shuffle=False):
    # Build a dictionary mapping from subject_id to list of indices
    subject_indices = defaultdict(list)
    for i in range(len(dataset)):
        sid = dataset.data_file['subject_id'][i]
        if isinstance(sid, bytes):
            sid = sid.decode("utf-8")  # convert from bytes to string
        subject_indices[sid].append(i)
    
    # Create dataloaders for each subject
    dataloader_dict = {}
    for sid, indices in subject_indices.items():
        subset = Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,drop_last=False)
        dataloader_dict[sid] = dataloader

    return dataloader_dict