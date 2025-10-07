import torch

def bbox_iou_xywh_xywh(pred, target, eps=1e-7):
    # pred/target: [...,4] (cx,cy,w,h) in pixels
    # Convert to xyxy
    px1 = pred[..., 0] - pred[..., 2] / 2
    py1 = pred[..., 1] - pred[..., 3] / 2
    px2 = pred[..., 0] + pred[..., 2] / 2
    py2 = pred[..., 1] + pred[..., 3] / 2

    tx1 = target[..., 0] - target[..., 2] / 2
    ty1 = target[..., 1] - target[..., 3] / 2
    tx2 = target[..., 0] + target[..., 2] / 2
    ty2 = target[..., 1] + target[..., 3] / 2

    # Intersection
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih

    # Union
    p_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    t_area = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = p_area + t_area - inter + eps
    iou = inter / union
    return iou, (px1, py1, px2, py2, tx1, ty1, tx2, ty2)

def bbox_ciou_xywh(pred, target, eps=1e-7):
    """
    CIoU loss (Zheng et al. 2020) on (cx,cy,w,h) in pixels.
    Returns (1 - CIoU).
    """
    iou, (px1, py1, px2, py2, tx1, ty1, tx2, ty2) = bbox_iou_xywh_xywh(pred, target, eps=eps)

    # center distance
    pcx = pred[..., 0]; pcy = pred[..., 1]
    tcx = target[..., 0]; tcy = target[..., 1]
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    # enclosing box
    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)
    ex2 = torch.max(px2, tx2)
    ey2 = torch.max(py2, ty2)
    cw = (ex2 - ex1).clamp(min=0)
    ch = (ey2 - ey1).clamp(min=0)
    c2 = cw ** 2 + ch ** 2 + eps

    # aspect ratio term
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(target[..., 2] / (target[..., 3] + eps)) - torch.atan(pred[..., 2] / (pred[..., 3] + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    ciou = iou - (rho2 / c2) - alpha * v
    return (1 - ciou).clamp(min=0)