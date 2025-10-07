from __future__ import annotations
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import nms

from .backbone_mnv3 import MobileNetV3Backbone
# from .yolo_head import YoloHeadSingle
from yolo.utils.nms import batched_nms


class MiniYoloDetector(nn.Module):
    """
    Minimal YOLO-style detector with:
      - MobileNetV3-Small backbone (stride 8)
      - Single-scale head at stride=8
    API:
      forward(images) -> dict {"p3": raw_pred}
      decode(raw_pred, img_size) -> (cx,cy,w,h,obj,cls) tensors in *pixels*
      postprocess(images, conf_thres, nms_thres) -> per-image numpy outputs
    """
    def __init__(self, num_classes: int, img_size: int = 640, head_channels: int = 64, act: str = "silu"):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = int(img_size)
        self.stride = 8  # single scale
        # Backbone
        self.backbone = MobileNetV3Backbone()
        # Head
        #self.head = YoloHeadSingle(self.backbone.out_channels, num_classes, hid=head_channels, act=act)

        # cached grid for decode
        self._grid_hw = None
        self.register_buffer("_grid_x", torch.zeros(1), persistent=False)
        self.register_buffer("_grid_y", torch.zeros(1), persistent=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.backbone(x)        # [B, C, Hs, Ws]
        p = self.head(f)            # [B, 5+C, Hs, Ws]
        return {"p3": p}

    @torch.no_grad()
    def _make_grid(self, device: torch.device, Hs: int, Ws: int):
        if self._grid_hw == (Hs, Ws) and self._grid_x.device == device:
            return self._grid_x, self._grid_y
        gy, gx = torch.meshgrid(torch.arange(Hs, device=device), torch.arange(Ws, device=device), indexing="ij")
        self._grid_x = gx.float()[None, ..., None]  # [1,Hs,Ws,1]
        self._grid_y = gy.float()[None, ..., None]
        self._grid_hw = (Hs, Ws)
        return self._grid_x, self._grid_y

    def decode(self, p):
        """
        p: logits from head, shape [B, 5+C, Hs, Ws] OR already split.
        Returns cx, cy, w, h, obj_logits, cls_logits in **pixels**.
        """
        B, K, Hs, Ws = p.shape
        tx = p[:, 0:1, ...]
        ty = p[:, 1:2, ...]
        tw = p[:, 2:3, ...]
        th = p[:, 3:4, ...]
        tobj = p[:, 4:5, ...]
        tcls = p[:, 5:,   ...]

        device = p.device
        ys = torch.arange(Hs, device=device).view(1,1,Hs,1).expand(B,1,Hs,Ws)
        xs = torch.arange(Ws, device=device).view(1,1,1,Ws).expand(B,1,Hs,Ws)

        cx_cell = (tx.sigmoid() + xs)      # [B,1,Hs,Ws]
        cy_cell = (ty.sigmoid() + ys)
        w_cell  = tw.exp()
        h_cell  = th.exp()

        # permute to [B,Hs,Ws,1 or C]
        cx_cell = cx_cell.permute(0,2,3,1)
        cy_cell = cy_cell.permute(0,2,3,1)
        w_cell  = w_cell.permute(0,2,3,1)
        h_cell  = h_cell.permute(0,2,3,1)
        obj = tobj.permute(0,2,3,1)        # logits
        cls = tcls.permute(0,2,3,1)        # logits
        return cx_cell, cy_cell, w_cell, h_cell, obj, cls
    
    
    def decode_cells(self, p):
        """
        p: [B, 5+C, Hs, Ws] logits
        returns: cx_cell, cy_cell, w_cell, h_cell, obj_logits, cls_logits
                with cx/cy in cell units and w/h in cell units (no stride)
        """
        B, K, Hs, Ws = p.shape
        tx = p[:, 0:1]; ty = p[:, 1:2]; tw = p[:, 2:3]; th = p[:, 3:4]
        tobj = p[:, 4:5]; tcls = p[:, 5:]

        device = p.device
        ys = torch.arange(Hs, device=device).view(1,1,Hs,1).expand(B,1,Hs,Ws)
        xs = torch.arange(Ws, device=device).view(1,1,1,Ws).expand(B,1,Hs,Ws)

        cx_cell = (tx.sigmoid() + xs).permute(0,2,3,1)  # [B,Hs,Ws,1]
        cy_cell = (ty.sigmoid() + ys).permute(0,2,3,1)
        w_cell  = (tw.exp()          ).permute(0,2,3,1)
        h_cell  = (th.exp()          ).permute(0,2,3,1)
        obj     =  tobj.permute(0,2,3,1)               # logits
        cls     =  tcls.permute(0,2,3,1)               # logits
        return cx_cell, cy_cell, w_cell, h_cell, obj, cls
    
    def decode_pixels(self, p, Himg: int, Wimg: int):
        """
        Convert cell units to pixel units using dynamic stride derived from the feature map.
        """
        cx_cell, cy_cell, w_cell, h_cell, obj, cls = self.decode_cells(p)
        _, Hs, Ws, _ = cx_cell.shape
        sx = Wimg / float(Ws); sy = Himg / float(Hs)
        cx = cx_cell * sx; cy = cy_cell * sy; w = w_cell * sx; h = h_cell * sy
        return cx, cy, w, h, obj, cls

    @torch.no_grad()
    def postprocess(self, images, conf_thres=0.25, nms_thres=0.6,
                pre_nms_topk=300, max_det=100):
        """
        Returns a list (per image) of (boxes_xyxy[N,4], scores[N], labels[N]) as NumPy.
        All coordinates are in pixel space of the network input (e.g., 640x640).
        """
        self.eval()

        # forward + decode
        preds = self.forward(images)["p3"]
        _, _, Himg, Wimg = images.shape
        cx, cy, w, h, obj, cls = self.decode_pixels(preds, Himg, Wimg)

        if not hasattr(self, "_debug_once"):
            self._debug_once = True
            print("[POSTPROC-DEBUG] cx range:", float(cx.min()), float(cx.max()))
            print("[POSTPROC-DEBUG] cy range:", float(cy.min()), float(cy.max()))
            print("[POSTPROC-DEBUG] w range:", float(w.min()), float(w.max()))
            print("[POSTPROC-DEBUG] h range:", float(h.min()), float(h.max()))
        B, Hs, Ws, _ = cx.shape

        outputs = []
        for b in range(B):
            # --- per-class scores (YOLO-style) ---
            obj_b    = obj[b, ..., 0]                 # [Hs, Ws]
            cls_b    = cls[b]                         # [Hs, Ws, C]
            obj_prob = obj_b.sigmoid()                # [Hs, Ws]
            cls_prob = torch.softmax(cls_b, dim=-1)                # [Hs, Ws, C]
            scores_3d = obj_prob[..., None] * cls_prob   # [Hs, Ws, C]

            # threshold per class
            mask_3d = scores_3d > conf_thres          # [Hs, Ws, C]
            if not mask_3d.any():
                outputs.append((
                    np.zeros((0, 4), np.float32),
                    np.zeros((0,),   np.float32),
                    np.zeros((0,),   np.int32),
                ))
                continue

            # gather indices of (iy, ix, c) that pass the mask
            inds = mask_3d.nonzero(as_tuple=False)    # [N, 3] -> (iy, ix, c)
            iy = inds[:, 0]; ix = inds[:, 1]; c = inds[:, 2]

            # gather boxes & scores (no flatten/view needed)
            cx_b = cx[b, iy, ix, 0]
            cy_b = cy[b, iy, ix, 0]
            w_b  =  w[b, iy, ix, 0]
            h_b  =  h[b, iy, ix, 0]
            sco_b = scores_3d[iy, ix, c]
            lab_b = c.to(torch.int64)

            # cxcywh -> xyxy
            x1 = cx_b - w_b / 2
            y1 = cy_b - h_b / 2
            x2 = cx_b + w_b / 2
            y2 = cy_b + h_b / 2
            boxes  = torch.stack([x1, y1, x2, y2], dim=1)  # [N,4]
            scores = sco_b
            labels = lab_b

            # clip to image bounds (just in case)
            if hasattr(images, "shape"):
                _, _, H, W = images.shape
                boxes[:, 0].clamp_(0, W - 1)
                boxes[:, 1].clamp_(0, H - 1)
                boxes[:, 2].clamp_(0, W - 1)
                boxes[:, 3].clamp_(0, H - 1)

            N = scores.numel()
            if N == 0:
                outputs.append((
                    np.zeros((0, 4), np.float32),
                    np.zeros((0,),   np.float32),
                    np.zeros((0,),   np.int32),
                ))
                continue

            C = int(cls.shape[-1])
            Hs, Ws = int(cx.shape[1]), int(cx.shape[2])
            keep_k = min(pre_nms_topk, Hs*Ws*C)  # e.g., pass pre_nms_topk=9600 for 40x40x6
            if N > keep_k:
                v, idx = torch.topk(scores, keep_k)
                boxes  = boxes[idx]
                scores = scores[idx]
                labels = labels[idx]

            keep = batched_nms(boxes, scores, labels, nms_thres)
            boxes  = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # 3) Post-NMS cap
            if scores.numel() > max_det:
                v, idx = torch.topk(scores, max_det)
                boxes  = boxes[idx]
                scores = scores[idx]
                labels = labels[idx]

            # 4) To NumPy
            outputs.append((
                boxes.detach().cpu().numpy().astype(np.float32),
                scores.detach().cpu().numpy().astype(np.float32),
                labels.detach().cpu().numpy().astype(np.int32),
            ))

        return outputs


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """
    Factory used by train.py
    """
    num_classes = int(cfg["num_classes"])
    img_size = int(cfg["img_size"])
    head_channels = int(cfg.get("head_channels", 64))
    act = cfg.get("act", "silu")
    model = MiniYoloDetector(num_classes=num_classes, img_size=img_size, head_channels=head_channels, act=act)
    return model