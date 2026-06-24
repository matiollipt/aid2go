import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, ia_weights=None):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if ia_weights is not None:
            # Scale positive class loss using (1.0 + IA) to ensure specific GO terms are prioritized
            weight = torch.ones_like(targets)
            pos_mask = (targets == 1)
            weight[pos_mask] = 1.0 + ia_weights[pos_mask]
            focal_loss = focal_loss * weight
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_loss_fn(name: str):
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "focal":
        return WeightedFocalLoss()
    else:
        raise ValueError(f"Unknown loss {name}")
