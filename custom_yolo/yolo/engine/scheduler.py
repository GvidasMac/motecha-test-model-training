import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineWithWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, max_iters, min_lr_ratio=0.01, last_epoch=-1):
        self.warmup_iters = int(warmup_iters)
        self.max_iters = int(max_iters)
        self.min_lr_ratio = float(min_lr_ratio)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        it = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if it < self.warmup_iters:
                lr = base_lr * float(it) / max(1, self.warmup_iters)
            else:
                t = (it - self.warmup_iters) / max(1, self.max_iters - self.warmup_iters)
                cos = 0.5 * (1 + math.cos(math.pi * t))
                lr = base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cos)
            lrs.append(lr)
        return lrs

def build_scheduler(optimizer, cfg, iters_per_epoch):
    sched = (cfg.get("scheduler", "cosine") or "cosine").lower()
    warmup_epochs = int(cfg.get("warmup_epochs", 3))
    total_epochs = int(cfg["epochs"])
    warmup_iters = warmup_epochs * iters_per_epoch
    max_iters = total_epochs * iters_per_epoch
    if sched == "cosine":
        return CosineWithWarmup(optimizer, warmup_iters=warmup_iters, max_iters=max_iters, min_lr_ratio=0.05)
    return None