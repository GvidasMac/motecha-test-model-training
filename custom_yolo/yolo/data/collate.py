import torch

def collate_fn(batch):
    imgs, targs = list(zip(*batch))
    imgs = torch.stack(imgs, 0)
    return imgs, list(targs)