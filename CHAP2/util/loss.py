import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """
    Focal loss for binary classification with logits.
    
    Args
    ----
    alpha : float, default 0.25  
        Weight for the positive class. Set to None to disable class balancing.
    gamma : float, default 2.0  
        Focusing parameter that down-weights easy examples.
    reduction : str, default "mean"  
        "none", "mean", or "sum", following PyTorch conventions.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs : logits of shape (N, *)  
        targets : binary labels of the same shape, values 0 or 1
        """
        # Binary cross entropy with logits, element-wise
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        # Convert to probabilities
        p_t = torch.exp(-bce_loss)           # p_t = sigmoid(logits) for correct class
        # Focal scaling
        focal_term = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_term * bce_loss
        else:
            focal_loss = focal_term * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
