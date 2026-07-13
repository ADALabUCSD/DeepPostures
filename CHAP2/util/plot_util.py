import matplotlib.pyplot as plt
import numpy as np

def min_max_normalize(data):
    """Min-max normalize a numpy array."""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def mean_std_normalize(data):
    """Mean and standard deviation normalize a numpy array."""
    return (data - np.mean(data)) / np.std(data)


def plot_masked_series(mask, predict_series, target_series, 
                       variate_labels=None, title=None):
    """
    Plots the unpatched series, original series, and highlights the masked regions.
    
    Parameters:
    - mask (torch.Tensor): The mask tensor of shape (6, 50).
    - predict_series (torch.Tensor): The predicted series tensor of shape (6, 200).
    - target_series (torch.Tensor): The original series tensor of shape (6, 200).
    - variate_labels (list, optional): List of labels for each variate. Default is None.
    - title (str, optional): Title for the entire plot. Default is None.
    """
    # Convert tensors to numpy arrays
    mask_np = mask.numpy().squeeze()
    predict_series = predict_series.numpy().squeeze()
    target_series = target_series.numpy().squeeze()
    
    # Mean-std normalization
    predict_series = mean_std_normalize(predict_series)
    target_series = mean_std_normalize(target_series)

    print('mask_np', mask_np.shape)
    print('predict_series', predict_series.shape)
    print('target_series', target_series.shape)

    # Patch width
    num_patches = mask_np.shape[1]  # Number of patches (should be 50)
    patch_width = predict_series.shape[1] // num_patches

    # Create subplots: 2 rows, 3 columns
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Flatten axs for easier indexing
    axs = axs.flatten()

    # Default variate labels if not provided
    if variate_labels is None:
        variate_labels = ["Gyr_X", "Gyr_Y", "Gyr_Z", "Acc_X", "Acc_Y", "Acc_Z"]

    # Calculate global min and max for y-axis scaling across all plots
    global_min = min(np.min(predict_series), np.min(target_series))
    global_max = max(np.max(predict_series), np.max(target_series))

    # Plot each of the 6 plots for predict_series, target_series, and highlight the masked area
    for i in range(6):
        axs[i].plot(predict_series[i, :], label='Predicted Series', color='cornflowerblue',linewidth=3)
        axs[i].plot(target_series[i, :], label='Original Series', color='tomato',linewidth=3)

        # Set the y-axis limits to a common scale
        axs[i].set_ylim([global_min, global_max])

        # Highlight masked regions
        for j, mask_value in enumerate(mask_np[i]):
            if mask_value > 0:
                start = j * patch_width
                end = (j + 1) * patch_width
                axs[i].fill_betweenx(
                    [global_min, global_max], 
                    start, end, 
                    color='gray', 
                    alpha=0.5, 
                    label='Masked Area' if j == 0 else ""
                )

        axs[i].set_title(variate_labels[i], fontsize=16)
        axs[i].set_xticks([])  # Remove numerical x-axis ticks
        axs[i].set_yticks([])  # Remove numerical x-axis ticks

    # Set only one legend for all subplots
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=16)

    # Set the overall title for the plot
    if title:
        fig.suptitle(title, fontsize=18)

    # Set x-axis label for the bottom row
    axs[4].set_xlabel('Time Step', fontsize=14)

    # Set y-axis labels font size for the left column
    # for i in [0, 3]:
    #     axs[i].set_ylabel('Normalized Value', fontsize=14)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Show the plot
    plt.show()
