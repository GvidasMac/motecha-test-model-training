from __future__ import annotations
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_iou import bbox_ciou_xywh, bbox_iou_xywh_xywh
from .focal_bce import bce_logits, focal_bce_logits

class YoloLoss(nn.Module):
    """
    Combines:
      - Box loss: CIoU on positive cells
      - Obj loss: BCE (or focal) on objectness map
      - Cls loss: BCE (or focal) on class logits, positive cells only
    Uses an assigner to create target maps per image.
    """
    def __init__(self, cfg: Dict[str, Any], assigner, num_classes, class_weights=None, box_weight=1.0, obj_weight=1.0):
        super().__init__()
        self.cfg = cfg
        self.assigner = assigner
        self.num_classes = int(cfg["num_classes"])
        self.stride = int(cfg["strides"][0])
        self.box_weight = float(cfg.get("box_weight", 1.0))
        self.obj_pos_weight = float(cfg.get("obj_pos_weight", 1.0))
        self.cls_pos_weight = float(cfg.get("cls_pos_weight", 1.0))
        self.use_focal = False # (cfg.get("cls_loss", "bce") == "focal") or (cfg.get("obj_loss", "bce") == "focal")
        super().__init__()
        self.num_classes = num_classes
        self.box_weight  = box_weight
        self.obj_weight  = obj_weight
        # optional per-class weights for CE
        self.ce_weight = None
        if class_weights is not None:
            self.ce_weight = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, preds: Dict[str, torch.Tensor],
            targets_list: List[torch.Tensor],
            img_size: int):
        # Single scale "p3"
        p = preds["p3"]  # [B, 5+C, Hs, Ws]
        B, C, Hs, Ws = p.shape
        device = p.device

        # logits split
        x = p.permute(0, 2, 3, 1)  # [B,Hs,Ws,5+C]
        tx, ty, tw, th, tobj = torch.split(x[..., :5], 1, dim=-1)
        tcls = x[..., 5:]

        # ----- DYNAMIC STRIDE (match postprocess) -----
        # If your net input is square 640, this is equivalent to sx=sy=img_size/Ws or Hs
        sx = float(img_size) / float(Ws)
        sy = float(img_size) / float(Hs)

        # grid in cell units
        gy, gx = torch.meshgrid(
            torch.arange(Hs, device=device),
            torch.arange(Ws, device=device),
            indexing="ij"
        )
        gx = gx[None, ..., None].float()  # [1,Hs,Ws,1]
        gy = gy[None, ..., None].float()

        # ----- DECODE TO PIXELS (same as postprocess) -----
        pred_cx = (torch.sigmoid(tx) + gx) * sx
        pred_cy = (torch.sigmoid(ty) + gy) * sy
        pred_w  = torch.exp(tw) * sx
        pred_h  = torch.exp(th) * sy
        pred_box = torch.cat([pred_cx, pred_cy, pred_w, pred_h], dim=-1)  # [B,Hs,Ws,4]

        if not hasattr(self, "_debug_once"):
            self._debug_once = True
            print("[LOSS-DEBUG] pred_box cx range:", float(pred_cx.min()), float(pred_cx.max()))
            print("[LOSS-DEBUG] pred_box cy range:", float(pred_cy.min()), float(pred_cy.max()))
            print("[LOSS-DEBUG] pred_box w range:", float(pred_w.min()), float(pred_w.max()))
            print("[LOSS-DEBUG] pred_box h range:", float(pred_h.min()), float(pred_h.max()))

        # ----- ASSIGNER (needs to be in sync; see next section) -----
        assigns = self.assigner(targets_list, (Hs, Ws), (img_size, img_size))

        obj_t = torch.stack([a["obj"] for a in assigns], dim=0)  # [B,Hs,Ws,1]
        obj_logit = tobj  # logits

        # ----- POSITIVE MASK & TARGET BOXES -----
        box_t = torch.stack([a["box"] for a in assigns], dim=0)   # [B,Hs,Ws,4] in PIXELS
        pos_m = torch.stack([a["mask"] for a in assigns], dim=0)  # [B,Hs,Ws,1]
        pos_mask = (pos_m[..., 0] > 0.5)                          # [B,Hs,Ws] bool

        # ----- IoU-AWARE OBJECTNESS TARGET -----
        # obj_t has IoU on positives, 0 on negatives
        with torch.no_grad():
            obj_t = torch.zeros_like(tobj)                        # [B,Hs,Ws,1], logits shape
            if pos_mask.any():
                # compute plain IoU between predicted and target boxes (in PIXELS)
                iou_pos, _ = bbox_iou_xywh_xywh(
                    pred_box[pos_mask],   # [Npos,4] (cx,cy,w,h)
                    box_t[pos_mask]       # [Npos,4]
                )
                obj_t[pos_mask] = iou_pos.unsqueeze(-1)          # [Npos,1]

        obj_logit = tobj  # logits

        # Objectness loss (focal BCE recommended)
        if getattr(self, "use_focal", True):
            l_obj = focal_bce_logits(obj_logit, obj_t, gamma=2.0, alpha=0.25, reduction="mean")
        else:
            l_obj = F.binary_cross_entropy_with_logits(obj_logit, obj_t, reduction="mean")

        # ----- BOX LOSS (CIoU) on positives -----
        if pos_mask.any():
            l_box = bbox_ciou_xywh(pred_box[pos_mask], box_t[pos_mask]).mean()
        else:
            l_box = pred_box.sum() * 0.0

        # ----- CLASS LOSS on positives (Cross-Entropy with indices) -----
        # Build integer label targets per positive cell; -100 = ignore
        cls_idx_t = torch.full((B, Hs, Ws), -100, device=device, dtype=torch.long)

        for b in range(B):
            pos_idx = torch.nonzero(pos_mask[b], as_tuple=False)  # [Npos, 2] (iy, ix)
            if pos_idx.numel() == 0:
                continue
            tgt = targets_list[b].to(device)  # [Ngt,5] [cls, cx, cy, w, h] in PIXELS
            if tgt.numel() == 0:
                continue

            # center of the cell in pixels
            cell_px = (pos_idx[:, 1].float() + 0.5) * sx
            cell_py = (pos_idx[:, 0].float() + 0.5) * sy

            gt_cx = tgt[:, 1][None, :]  # [1, Ngt]
            gt_cy = tgt[:, 2][None, :]  # [1, Ngt]
            d2 = (cell_px[:, None] - gt_cx) ** 2 + (cell_py[:, None] - gt_cy) ** 2
            j = torch.argmin(d2, dim=1)  # nearest GT index per positive cell
            cls_ids = tgt[j, 0].long()   # [Npos]
            cls_idx_t[b, pos_idx[:, 0], pos_idx[:, 1]] = cls_ids

        # Cross-entropy on raw logits (no sigmoid), only on positives
        if pos_mask.any():
            logits_pos = tcls[pos_mask]              # [Npos, C]
            targets_pos = cls_idx_t[pos_mask]
            w = self.ce_weight.to(logits_pos.device) if self.ce_weight is not None else None
            # tcls is [B,Hs,Ws,C]; select positives → [Npos, C]
            l_cls = F.cross_entropy(
                logits_pos,              # logits at positives
                targets_pos,           # int labels
                ignore_index=-100,
                label_smoothing=0.05,
                weight=w,
                reduction="mean",
            )
        else:
            l_cls = tcls.sum() * 0.0

        loss = (1.0 * l_box) + (2.0 * l_obj) + (1.0 * l_cls)
        loss = self.box_weight * l_box + l_obj + l_cls
        parts = {"l_box": float(l_box.detach()), "l_obj": float(l_obj.detach()), "l_cls": float(l_cls.detach())}
        return loss, parts