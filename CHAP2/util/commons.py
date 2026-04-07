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
        assert datetime.strptime(fnames[i + 1], "%Y-%m-%d").date() - datetime.strptime(
            fnames[i], "%Y-%m-%d"
        ).date() == timedelta(days=1)

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


def window_generator(data_root, win_size_10s, subject_ids):
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
                        yield np.stack(x_window, axis=0), np.stack(y_window, axis=0)
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
):
    """
    Process data and get dataloaders for subject
    """

    train_dataloader = None
    valid_dataloader = None
    test_dataloader = None
    # https://github.com/pytorch/ignite/issues/1076#issuecomment-829191233
    if train_subjects:
        train_data = IterDatasetDist(
            window_generator,
            rank,
            world_size,
            pre_processed_dir,
            bi_lstm_win_size,
            train_subjects,
        )
        train_dataloader = DataLoader(
            train_data, batch_size=batch_size, pin_memory=True
        )

    if valid_subjects:
        valid_data = IterDatasetDist(
            window_generator,
            rank,
            world_size,
            pre_processed_dir,
            bi_lstm_win_size,
            valid_subjects,
        )
        valid_dataloader = DataLoader(
            valid_data, batch_size=batch_size, pin_memory=True
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
            train_data, batch_size=batch_size, pin_memory=True
        )

    if valid_subjects:
        valid_data = IterDataset(
            window_generator, pre_processed_dir, bi_lstm_win_size, valid_subjects
        )
        valid_dataloader = DataLoader(
            valid_data, batch_size=batch_size, pin_memory=True
        )
    if test_subjects:
        test_data = IterDataset(
            window_generator, pre_processed_dir, bi_lstm_win_size, test_subjects
        )
        test_dataloader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

    return train_dataloader, valid_dataloader, test_dataloader