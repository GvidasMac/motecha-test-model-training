import torch
import torch.nn.functional as F

def bce_logits(input, target, pos_weight=1.0, reduction="mean"):
    return F.binary_cross_entropy_with_logits(input, target, pos_weight=torch.tensor(pos_weight, device=input.device), reduction=reduction)

def focal_bce_logits(input, target, gamma=2.0, alpha=0.25, reduction="mean"):
    # Standard focal on logits
    p = torch.sigmoid(input)
    ce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
    p_t = p*target + (1-p)*(1-target)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha is not None:
        alpha_t = alpha*target + (1-alpha)*(1-target)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss