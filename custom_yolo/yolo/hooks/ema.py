import torch
import torch.nn as nn
import copy
import math

class ModelEMA:
    """
    Exponential Moving Average of model weights.
    Call ema.update(model) after each optimizer step.
    """
    

    def __init__(self, model: nn.Module, decay=0.9999, tau=2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = 0

    def update(self, model: nn.Module):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= (1.0 - d)
                v += d * msd[k].detach()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        for k, v in model.__dict__.items():
            if (len(include) and k not in include) or k in exclude:
                continue
            setattr(self.ema, k, v)