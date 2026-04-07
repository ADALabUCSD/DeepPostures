import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from scipy.signal import resample
from transforms3d.axangles import axangle2mat
from einops import rearrange

def rotation_axis(sample):
    """

    Rotate the input sample along a random axis by a random angle.
    Modified from: OxWearables. (2022). 
    ssl-wearables: Self-supervised learning for wearable sensor data. 
    GitHub repository. https://github.com/OxWearables/ssl-wearables/blob/main/sslearning/data/datautils.py
    
    Args:
        sample (numpy.ndarray): Input sample of shape (T, C), where T is the number of time steps and C is the number of channels.
    
    Returns:
        numpy.ndarray: Rotated sample of the same shape as input.
    """
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    axis = np.random.uniform(low=-1, high=1, size=sample.shape[1])
    rotation_matrix = axangle2mat(axis, angle)
    sample = np.matmul(sample, rotation_matrix)

    return sample

def channel_permute(sample):
    """
    Permute the channels of the input sample.
    
    Args:
        sample (numpy.ndarray): Input sample of shape (T, C), where T is the number of time steps and C is the number of channels.
    
    Returns:
        numpy.ndarray: Sample with permuted channels.
    """
    perm = np.random.permutation(sample.shape[1])
    return sample[:, perm]

def scaling(sample, scale_factor=0.1):
    """
    Scale the input sample by a random factor.

    Args:
        sample (numpy.ndarray): Input sample of shape (T, C), where T is the number of time steps and C is the number of channels.
        scale_factor (float): Factor by which to scale the sample.

    Returns:
        numpy.ndarray: Scaled sample.
    """
    sample *= np.random.normal(loc=1.0, scale=scale_factor)
    return sample

def jittering(sample, noise_level=0.05):
    """
    Add Gaussian noise to the input sample.
    
    Args:
        sample (numpy.ndarray): Input sample of shape (T, C), where T is the number of time steps and C is the number of channels.
        noise_level (float): Standard deviation of the Gaussian noise to be added.
    
    Returns:
        numpy.ndarray: Sample with added Gaussian noise.
    """
    noise = np.random.normal(loc=0.0, scale=noise_level, size=sample.shape)
    return sample + noise

from scipy.interpolate import CubicSpline
import numpy as np

def DistortTimesteps(X, sigma=0.2):
    # X shape: (length, nvar)
    tt = GenerateRandomCurves(X, sigma)
    tt_cum = np.cumsum(tt, axis=0)

    # scale time to match original length
    t_scale = (X.shape[0] - 1) / tt_cum[-1]
    tt_cum = tt_cum * t_scale  # element-wise scale for each variable
    return tt_cum

def GenerateRandomCurves(X, sigma=0.2, knot=4):
    # X shape: (length, nvar)
    length, nvar = X.shape
    xx = np.linspace(0, length - 1, num=knot + 2)  # shape: (knot+2,)
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, nvar))  # shape: (knot+2, nvar)
    
    x_range = np.arange(length)
    tt = np.zeros((length, nvar))
    for i in range(nvar):
        cs = CubicSpline(xx, yy[:, i])
        tt[:, i] = cs(x_range)
    return tt

def DA_TimeWarp(X, sigma=0.2):
    # X shape: (length, nvar)
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros_like(X)
    x_range = np.arange(X.shape[0])
    for i in range(X.shape[1]):
        X_new[:, i] = np.interp(x_range, tt_new[:, i], X[:, i])
    return X_new

def time_warp(sample, sigma=0.2):

    sample = DA_TimeWarp(sample, sigma=sigma)

    return sample

def data_aug(x):
    """
    Input:
        x: numpy array of shape (T, C), typically (100, 3)
           T is the number of time steps, C is the number of channels
    Output:
        x_aug: numpy array of the same shape with augmentations applied

    Augmentations [1]:
        - Gaussian noise (jittering)
        - Global scaling
        - Channel permutation: invariant to the order of channels, because different device manafacturers may have different channel orders
        - Axis flipping: we want the model invariant to subject that wear the device differently
    
    Notes: 
        -Should not apply normalization for activity data [2]


    Reference:
        [1] https://shamilmamedov.com/blog/2023/da-time-series/
        [2] https://www.mdpi.com/1999-5903/12/11/194

    """

    x = x.astype(np.float32).copy()

    x = rotation_axis(x) 
    x = channel_permute(x)
    x = jittering(x)
    x = scaling(x)
    #x = time_warp(x)

    return x

import h5py
class iWatch(Dataset):
    def __init__(self, 
                 root='/niddk-data-central/iWatch/pre_processed_long_seg/H',
                 set_type='train',
                 transform=None,
                 std_sampling=False,
                 subset_ratio=1.0):
        
        self.file_path = os.path.join(root, f"10s_{set_type}.h5")
        self.data_file = h5py.File(self.file_path, 'r')
        self.x_data = self.data_file['x']       # shape: (N,window, 100, 3)
        self.y_data = self.data_file['y']       # shape: (N, window)
        self.std_sampling = std_sampling

        if self.std_sampling and 'std' in self.data_file:
            self.stds = self.data_file['std'][:] # materialized it.  (BS, window)
            self.stds = self.stds.mean(axis=1)
        else:
            self.stds = None

        self.timestamp = self.data_file['timestamp'] # shape: (N, window, )
        self.transform = transform
        self.subject_id = np.unique(self.data_file['subject_id'])
        
        self.indices = np.arange(len(self.y_data))
        
        # each subject should have 10% so the distrbution for each subject is the same as before
        if subset_ratio < 1.0:
            self.subject_id = self.data_file['subject_id']
            np.random.seed(42)
            final_indices = []
            subject_ids = np.unique(self.subject_id)
            for sid in subject_ids:
                subject_indices = np.where(self.subject_id[:] == sid)[0]
                num_samples = max(1, int(len(subject_indices) * subset_ratio))
                sampled = np.random.choice(subject_indices, num_samples, replace=False)
                final_indices.extend(sampled)

            self.indices = np.array(final_indices)
            
        if self.stds is not None: # FIXME: problematic when using subset_ratio
            self.indices_with_std = np.column_stack((self.indices, self.stds[self.indices])) #(Bs, 2)
        else:
            self.indices_with_std = None

    def resample_epoch(self):
        if self.std_sampling:
            self.indices = weighted_epoch_sample(self.indices_with_std)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        x = self.x_data[idx]  # shape: (42, 100, 3)
        y = self.y_data[idx]  # shape: (42,)
        timestamp = self.timestamp[idx]  # shape: (42,)
        
        if self.transform is not None:
            x_aug = x.reshape(-1, x.shape[-1]) # (4200,3)
            x_aug = self.transform(x_aug)
            x_aug = x_aug.reshape(x.shape[0], x.shape[1], -1)  # Reshape back to (42, 100, 3)

        else:
            x_aug = x.copy()

        x_aug = torch.from_numpy(x_aug).to(dtype=torch.float32) # (42, 100, 3)
        y = torch.tensor(y, dtype=torch.long)

        return x_aug, y, timestamp


def weighted_epoch_sample(indicies_with_std):
    """
    Weighted sample the windows that have most motion
    Args:
        data_with_std (np_array) of shape N x 2:
    Returns:
        sample_ides (np_array): indices of the sampled windows
    """
    # sample_len = 100
    indicies = indicies_with_std[:, 0]  # Get the indices
    std = indicies_with_std[:, 1]  # Get the std values

    sample_ides = np.random.choice(
        len(indicies), len(indicies), replace=True, p=std / np.sum(std)
    )

    return sample_ides


    
# Dataset for (BS, 100,3)
import h5py
from scipy.signal import resample
class iWatch_HDf5(Dataset):
    def __init__(self,
                 root='/niddk-data-central/iWatch/pre_processed_seg/H',
                 set_type='train',
                 transform=None,
                 subset=None,
                 target_sr=10):
        self.file_path = os.path.join(root, f"10s_{set_type}.h5")
        self.transform = transform
        # these will be set in the worker when first accessed
        self.h5_file = None
        self.x_data = None
        self.y_data = None
        self.subset= subset
        self.data_sr = 10
        self.target_sr = target_sr
    def _ensure_open(self):
        # called inside worker on first __getitem__
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, 'r')
            self.x_data = self.h5_file['x']
            self.y_data = self.h5_file['y']

    def __len__(self):
        # we open here if not already, so that len() works in main process
        self._ensure_open()
        if self.subset is not None:
            return self.subset
        
        return len(self.x_data)

    def __getitem__(self, idx):
        self._ensure_open()                     # open once per worker
        x = self.x_data[idx]                    # shape: (100, 3)
        # resample 
        if self.data_sr != self.target_sr:
            T, N = x.shape
            T_new = int(T * self.target_sr / self.data_sr)
            x = resample(x, T_new, axis=0)
        
        if self.transform is not None:
            x = self.transform(x) # shape: (100, 3)
            
        
        x = torch.from_numpy(x).permute(1, 0).float()  # shape: (3, 100)
        x = x.unsqueeze(0)                      # shape: (1, 3, 100)

        y = int(self.y_data[idx])               
        return x, torch.tensor(y, dtype=torch.long)

    def __del__(self):
        if getattr(self, 'h5_file', None) is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass


def simple_collate_fn(batch):
    clean_batch = []
    for x, y in batch:
        if torch.isnan(x).any() or torch.isinf(x).any():
            continue
        clean_batch.append((x, y))

    if len(clean_batch) == 0:
        return None  # or raise an error if needed

    xs, ys = zip(*clean_batch)
    return torch.stack(xs), torch.tensor(ys)

def flatten_collate_fn(batch):
    '''
    Each item:
    x: [win_size, 100, 3]
    y: [win_size]
    timestamp: [win_size]

    Output:
    x: [bs * win_size, 3, 100]
    y: [bs * win_size]
    timestamp: [bs * win_size]
    '''
    clean_x, clean_y, clean_timestamp = [], [], []

    for x, y, timestamp in batch:
        if torch.isnan(x).any() or torch.isinf(x).any():
            continue

        # x: (win_size, 100, 3) → (win_size,1, 3, 100)
        x = rearrange(x, 'w l c -> w 1 c l')  
        clean_x.append(x)
        clean_y.append(y)
        clean_timestamp.append(torch.as_tensor(timestamp))

    if len(clean_x) == 0:
        return None  # or raise error

    x = torch.cat(clean_x, dim=0)  # [bs * win_size, 1, 3, 100]
    y = torch.cat(clean_y, dim=0)  # [bs * win_size]
    timestamp = torch.cat(clean_timestamp, dim=0)  # [bs * win_size]

    return x, y, timestamp

def long_collate_fn(batch):
    '''
    Each item:
    x: [win_size, 100, 3]
    y: [win_size]
    timestamp: [win_size]

    Output:
    x: [bs, 3, win_size * 100]
    y: [bs * win_size]
    timestamp: [bs * win_size]
    '''
    clean_x, clean_y, clean_timestamp = [], [], []

    for x, y, timestamp in batch:
        if torch.isnan(x).any() or torch.isinf(x).any():
            continue

        # x: (win_size, 100, 3) → (3, win_size * 100)
        x = rearrange(x, 'w l c -> c (w l)')
        clean_x.append(x)
        clean_y.append(y)
        clean_timestamp.append(torch.as_tensor(timestamp))

    if len(clean_x) == 0:
        return None  # or raise error

    # Stack along batch dimension
    x = torch.stack(clean_x, dim=0).unsqueeze(1)  # [bs, 1, 3, win_size * 100]
    y = torch.cat(clean_y, dim=0)    # [bs * win_size]
    timestamp = torch.cat(clean_timestamp, dim=0)  # [bs * win_size]

    return x, y, timestamp


if __name__ == "__main__":
    print("Starting dataset loading and testing...")

    # Parameters
    batch_size = 4

    train_dataset = iWatch(set_type='train', transform=data_aug)
    test_dataset = iWatch(set_type='test', transform=None)
    val_dataset = iWatch(set_type='val', transform=None)

    # Print the length of the datasets
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("DataLoaders created successfully.")

    # Iterate through the train DataLoader
    print("Training DataLoader:")
    for i, (images, labels,_) in enumerate(train_loader):
        print(f"Batch {i + 1}:")
        print(f"Images shape: {images.shape}") # bs x nvar x 1 x L
        print(f"Labels shape: {labels.shape}") # bs 
        if i == 1:  # Just show first two batches
            break

    # Iterate through the test DataLoader
    print("\nTesting DataLoader:")
    for i, (images, labels,_) in enumerate(test_loader):
        print(f"Batch {i + 1}:")
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        if i == 1:  # Just show first two batches
            break