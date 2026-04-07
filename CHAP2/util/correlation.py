import torch
from scipy.linalg import sqrtm

def intercovariance(views):
    '''
    Input:  views(tensor): # batch_size, num_patch, patch_dim
    Output: covariance_matrix(tensor): # batch_size, num_patch, num_patch
    '''    

    # batch_size = views.shape[0]
    # num_p = views.shape[1]
    patch_dim = views.shape[2]

    # Center the data by subtracting the mean of each patch (important for covariance)
    views_centered = views - torch.mean(views, axis=2, keepdims=True) # xi - xbar
    # Compute covariance matrix using batch matrix multiplication
    covariance_matrix_batch = torch.matmul(views_centered, views_centered.transpose(0, 2, 1)) / (patch_dim - 1)
    
    return covariance_matrix_batch

    
def crosscovariance(view1, view2):
    '''
    Input:  view1(tensor): # bs x num_p1 x patch_dim
            view2(tensor): # bs x num_p2 x patch_dim
    Output: covariance_matrix: # num_p1 x num_p2
    '''    
    # num_p1 = view1.shape[0]
    # num_p2 = view2.shape[0]
    patch_dim = view1.shape[1]

    # Center the data by subtracting the mean of each patch (important for covariance)
    view1_centered = view1 - torch.mean(view1, axis=1, keepdims=True) # x_i - xbar
    view2_centered = view2 - torch.mean(view2, axis=1, keepdims=True) # y_i - ybar

    # Compute cross-covariance matrix using matrix multiplication
    covariance_matrix = torch.matmul(view1_centered, view2_centered.T) / (patch_dim - 1)#sample covariance
    
    return covariance_matrix


# # compile into one function
# def singular_value(mask_feats,unmask_feats):
#     uu = intercovariance(unmask_feats)
#     um = crosscovariance(unmask_feats,mask_feats)
#     mm = intercovariance(mask_feats)

#     optim = np.matmul(np.matmul(sqrtm(uu), um), sqrtm(mm))

#     _, S, _ = np.linalg.svd(optim)
#     singular = S[0]

#     return singular