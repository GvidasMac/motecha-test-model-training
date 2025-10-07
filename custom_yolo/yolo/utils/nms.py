import torch
from torchvision.ops import nms as tv_nms

def batched_nms(boxes: torch.Tensor,
                scores: torch.Tensor,
                labels: torch.Tensor,
                iou_th: float = 0.65) -> torch.Tensor:
    """
    Class-aware NMS: run NMS separately per class, then concatenate keeps.
    Returns indices into the original tensors.
    Works on CPU for MPS safety.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    # Always do NMS on CPU (safe for MPS), then map back
    boxes_cpu  = boxes.detach().cpu()
    scores_cpu = scores.detach().cpu()
    labels_cpu = labels.detach().cpu()

    keep_indices = []
    # unique classes present
    for c in labels_cpu.unique():
        m = labels_cpu == c
        if m.sum().item() == 0:
            continue
        k = tv_nms(boxes_cpu[m], scores_cpu[m], iou_th)
        # map local indices back to original indices
        global_idx = torch.nonzero(m, as_tuple=False).squeeze(1)[k]
        keep_indices.append(global_idx)

    if len(keep_indices) == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    keep = torch.cat(keep_indices, dim=0)

    # optional: sort kept indices by descending score so top-k later is stable
    _, order = torch.sort(scores_cpu[keep], descending=True)
    keep = keep[order].to(boxes.device)

    return keep