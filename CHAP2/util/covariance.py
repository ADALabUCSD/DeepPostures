# import numpy as np
# from scipy.linalg import sqrtm

# def intercovariance(views):    
#     if not isinstance(views, np.ndarray):
#         views = np.array(views, dtype=np.float32)

#     # Compute the pairwise covariance matrix
#     num_p = views.shape[0]
#     covariance_matrix = np.zeros((num_p, num_p))
    
#     for i in range(num_p):
#         for j in range(num_p):
#             covariance_matrix[i, j] = np.cov(views[i], views[j])[0, 1]
    
#     return covariance_matrix


# def crosscovariance(view1, view2):
#     if not isinstance(view1, np.ndarray):
#         view1 = np.array(view1, dtype=np.float32) # num_p1 x 768
#     if not isinstance(view2, np.ndarray):
#         view2 = np.array(view2, dtype=np.float32)  # num_p2 x 768

#     num_p1 = view1.shape[0]
#     num_p2 = view2.shape[0]
#     covariance_matrix = np.zeros((num_p1, num_p2))

#     for i in range(num_p1):
#         for j in range(num_p2):
#             covariance_matrix[i, j] = np.cov(view1[i], view2[j])[0, 1]

#     return covariance_matrix
    

# # compile into one function
# def singular_value(mask_feats,unmask_feats):
#     uu = intercovariance(unmask_feats)
#     um = crosscovariance(unmask_feats,mask_feats)
#     mm = intercovariance(mask_feats)

#     optim = np.matmul(np.matmul(sqrtm(uu), um), sqrtm(mm))

#     _, S, _ = np.linalg.svd(optim)
#     singular = S[0]

#     return singular


import numpy as np
from scipy.linalg import sqrtm

def intercovariance(views):
    views = np.asarray(views, dtype=np.float32)
    # Each row is treated as a variable with multiple observations.
    # Compute the mean for each row and center the data.
    centered = views - np.mean(views, axis=1, keepdims=True)
    # Compute the covariance matrix using matrix multiplication.
    covariance_matrix = centered @ centered.T / (views.shape[1] - 1)
    return covariance_matrix

def crosscovariance(view1, view2):
    view1 = np.asarray(view1, dtype=np.float32)
    view2 = np.asarray(view2, dtype=np.float32)
    # Compute the mean for each row and center the data.
    centered1 = view1 - np.mean(view1, axis=1, keepdims=True)
    centered2 = view2 - np.mean(view2, axis=1, keepdims=True)
    # Compute the cross-covariance matrix in a vectorized way.
    covariance_matrix = centered1 @ centered2.T / (view1.shape[1] - 1)
    return covariance_matrix

def singular_value(mask_feats, unmask_feats):
    uu = intercovariance(unmask_feats)
    um = crosscovariance(unmask_feats, mask_feats)
    mm = intercovariance(mask_feats)

    # Compute the product of the square roots of the covariance matrices and the cross-covariance.
    optim = sqrtm(uu) @ um @ sqrtm(mm)
    # Compute singular values using SVD.
    _, S, _ = np.linalg.svd(optim)
    singular = S[0]
    return singular



import numpy as np
import time
from util.covariance import singular_value
from tqdm import tqdm

def row_selection_matrix(R, n):
    S = np.zeros((len(R), n), dtype=float)
    for i, r_idx in enumerate(R):
        S[i, r_idx] = 1.0
    return S

def col_selection_matrix(C, n):
    T = np.zeros((n, len(C)), dtype=float)
    for j, c_idx in enumerate(C):
        T[c_idx, j] = 1.0
    return T

import torch
def row_selection_matrix(R, n):
    S = np.zeros((len(R), n), dtype=float)
    for i, r_idx in enumerate(R):
        S[i, r_idx] = 1.0
    return S

def col_selection_matrix(C, n):
    T = np.zeros((n, len(C)), dtype=float)
    for j, c_idx in enumerate(C):
        T[c_idx, j] = 1.0
    return T

def greedy_find_S_T(cov_matrix, mask_ratio=0.75):
    """
    Greedy algorithm to find the unmask set S
    
    Args:
        cov_matrix: Input square matrix (n x n)
        mask_ratio
        
    Returns:
        best_S: index of unmask set S

    """
    cov_matrix = cov_matrix.cpu().numpy()  # Ensure the computation is on CPU and convert to numpy array
    n = cov_matrix.shape[0]
    max_size = n - int(mask_ratio * n)

    R = []
    all_indices = set(range(n))
 
    for _ in range(max_size):
        best_candidate = None
        best_new_sv = 0
        for r in sorted(all_indices - set(R)):
            new_R = R + [r]
            Rc = sorted(list(all_indices - set(new_R)))
            # Extract submatrix 
            M = cov_matrix[np.ix_(new_R, Rc)]
            sv = np.linalg.norm(M, 2)  # spectral norm
            if sv > best_new_sv:
                best_new_sv = sv
                best_candidate = r
        R.append(best_candidate)

    return R

def spectral_mask(x,mask_ratio=0.75):
    """
    Spectral Masking
    
    Args:
        x: bs, num_patch, patch_size 
        mask_ratio
        
    Returns:
        x_masked: masked tensor (n x d)
        mask,
        ids_restored

    """
    N,L,D = x.shape
    len_keep = int(L * (1-mask_ratio))

    x_masked = torch.zeros(N,L,device=x.device)
    for i in range(N):
        cov_matrix = torch.abs(torch.cov(x[i])) # num_patch x num_patch
        unmask_idx = greedy_find_S_T(cov_matrix, mask_ratio)
        x_masked[i][unmask_idx] = 1

    ids_shuffle = torch.argsort(x_masked, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return ids_keep, mask, ids_restore