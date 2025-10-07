import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import amp as torch_amp

# Config / utils
from yolo.utils.config import load_config  # -> Dict
from yolo.utils.seed import set_seed
from yolo.utils.logging import get_logger, TqdmBar
from yolo.utils.metrics import EvalAP50  # simple AP50 evaluator
from yolo.utils.nms import batched_nms  # or a thin wrapper around torchvision.ops.nms
from yolo.utils.ap_eval import ap50_from_batches

# Data
from yolo.data.dataset_yolo_txt import YoloTxtDataset
from yolo.data.collate import collate_fn
from yolo.data.transforms import build_transforms  # returns train_tfms, val_tfms

# Model
from yolo.models.detector import build_model  # (cfg) -> nn.Module

# Loss / assigner
from yolo.losses.yolo_loss import YoloLoss
from yolo.assigners.naive_grid import NaiveGridAssigner

# Hooks
from yolo.hooks.ema import ModelEMA

# Scheduler
from yolo.engine.scheduler import build_scheduler  # (optimizer, cfg) -> torch.optim.lr_scheduler._LRScheduler

def human_bytes(num: int) -> str:
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}B"
        num /= 1024.0
    return f"{num:.1f}YB"


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_dataloaders(cfg: Dict[str, Any], device):
    train_tfms, val_tfms = build_transforms(cfg)
    pin_mem = (device.type == "cuda")

    train_set = YoloTxtDataset(
        root=cfg["data_root"],
        img_subdir=cfg["train_images"],
        lbl_subdir=cfg["train_labels"],
        classes=cfg["classes"],
        img_size=cfg["img_size"],
        transforms=train_tfms,
        split="train",
    )
    val_set = YoloTxtDataset(
        root=cfg["data_root"],
        img_subdir=cfg["val_images"],
        lbl_subdir=cfg["val_labels"],
        classes=cfg["classes"],
        img_size=cfg["img_size"],
        transforms=val_tfms,
        split="val",
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=pin_mem,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, cfg["batch_size"] // 2),
        shuffle=False,
        num_workers=max(1, cfg.get("num_workers", 4) // 2),
        pin_memory=pin_mem,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_set, val_set, train_loader, val_loader


def build_optimizer(cfg: Dict[str, Any], model: nn.Module):
    opt_name = cfg.get("optimizer", "adamw").lower()
    lr = float(cfg.get("lr", 3e-4))
    wd = float(cfg.get("weight_decay", 5e-4))
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return optimizer

def _safe_state_dict(m: nn.Module):
    # supports plain, DataParallel, and DDP
    if isinstance(m, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return m.module.state_dict()
    return m.state_dict()

def save_ckpt(path: Path, model: nn.Module, optimizer, epoch: int, scaler, cfg, ema: ModelEMA = None, metric_value=None):
    src_model = ema.ema if ema is not None else model
    ckpt = {
        "epoch": epoch,
        "model": _safe_state_dict(src_model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "cfg": cfg,
        "metric": metric_value,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(path))


def resume_if_possible(logger, model, optimizer, scaler, cfg, ema: ModelEMA = None):
    if not cfg.get("resume", False):
        return 0  # start epoch
    last_path = Path(cfg["save_dir"]) / "last.pt"
    if not last_path.exists():
        logger.info("No last.pt found to resume from.")
        return 0
    ckpt = torch.load(str(last_path), map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception:
        logger.warning("Could not load optimizer state; continuing fresh optimizer.")
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None:
        # initialize EMA from loaded model
        ema.ema.load_state_dict(ckpt["model"], strict=False)
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    logger.info(f"Resumed from {last_path} @ epoch {start_epoch-1}")
    return start_epoch


def validate(model, val_loader, device, cfg, logger, ema: ModelEMA = None):
    model_to_eval = ema.ema if (ema is not None) else model
    model_to_eval.eval()

    # thresholds / caps for eval
    conf_th = float(cfg.get("val_conf_thres", 0.05))
    nms_th  = float(cfg.get("val_nms_thres", 0.6))
    pre_nms_topk = int(cfg.get("val_pre_nms_topk", 300))   # pre-NMS keep
    max_det      = int(cfg.get("val_max_det", 100))        # post-NMS keep
    val_log_every = int(cfg.get("val_log_every", 50))      # reduce spam

    print(f'CONF_THRES= {conf_th}')

    # original evaluator (keep for comparison)
    ap50_eval = EvalAP50(
        class_names=cfg["classes"],
        iou_thresh=cfg["metrics"].get("iou_thresholds", [0.5])[0]
    )

    # smoke AP buffers
    smoke_preds = []  # list of (boxes_xyxy[N,4] float32, scores[N] float32, labels[N] int64)
    smoke_gts   = []  # list of (cls_ids[M] int32, boxes_xyxy[M,4] float32)

    with torch.no_grad():
        for ib, (images, targets) in enumerate(val_loader, 1):
            images = images.to(device, non_blocking=True)

            # NOTE: make sure your model.postprocess supports these kwargs
            if isinstance(model_to_eval, torch.nn.DataParallel):
                outputs = model_to_eval.module.postprocess(images,
                              conf_thres=conf_th, nms_thres=nms_th,
                              pre_nms_topk=pre_nms_topk, max_det=max_det)
            else:
                outputs = model_to_eval.postprocess(images,
                              conf_thres=conf_th, nms_thres=nms_th,
                              pre_nms_topk=pre_nms_topk, max_det=max_det)

            # light logging every N batches
            if ib % val_log_every == 0 or ib == 1:
                num_dets = [len(s) for (_, s, _) in outputs]
                if sum(num_dets) == 0:
                    logger.info(f"[VAL] b{ib}: no boxes after threshold (conf={conf_th})")
                else:
                    max_per_img = [float(s.max()) for (_, s, _) in outputs if len(s)]
                    mean_dets = sum(num_dets) / max(1, len(num_dets))
                    mean_max  = (sum(max_per_img)/len(max_per_img)) if max_per_img else 0.0
                    logger.info(f"[VAL] b{ib}: dets/img≈{mean_dets:.2f}, max score≈{mean_max:.3f}")

            # accumulate ORIGINAL metrics (expects same units as outputs)
            ap50_eval.add_batch(outputs, targets, img_size=cfg["img_size"])

            # ---- accumulate SMOKE AP (robust to tensor/numpy) ----
            def _to_np(x, dtype):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy().astype(dtype, copy=False)
                x = np.asarray(x)
                return x.astype(dtype, copy=False)

            # predictions
            for (b, s, l) in outputs:
                b_np = _to_np(b, np.float32)     # xyxy pixels
                s_np = _to_np(s, np.float32)
                l_np = _to_np(l, np.int64)
                smoke_preds.append((b_np, s_np, l_np))

            # GTs → xyxy pixels on the 640 canvas
            for t in targets:
                if t.numel() == 0:
                    smoke_gts.append((np.zeros((0,), np.int32), np.zeros((0,4), np.float32)))
                else:
                    t_np = t.detach().cpu().numpy().astype(np.float32)  # (N,5) [cls,cx,cy,w,h]
                    gx1 = t_np[:,1] - t_np[:,3]/2; gy1 = t_np[:,2] - t_np[:,4]/2
                    gx2 = t_np[:,1] + t_np[:,3]/2; gy2 = t_np[:,2] + t_np[:,4]/2
                    gxyxy = np.stack([gx1, gy1, gx2, gy2], 1).astype(np.float32)

                    # if they look normalized (<=1.5), scale to img_size
                    if float(gxyxy.max(initial=0.0)) <= 1.5:
                        gxyxy *= float(cfg["img_size"])

                    gcls = t_np[:,0].astype(np.int32)
                    smoke_gts.append((gcls, gxyxy))

            if ib == 1:
                # Take first image outputs
                b0, s0, l0 = outputs[0]  # boxes xyxy (np), scores (np), labels (np)
                # Build GT xyxy for first image from the transformed targets
                t0 = targets[0]  # Tensor [N,5] (cls,cx,cy,w,h) in PIXELS
                if hasattr(t0, "cpu"): t0 = t0.cpu().numpy()
                if t0.size:
                    gx1 = t0[:,1] - t0[:,3]/2; gy1 = t0[:,2] - t0[:,4]/2
                    gx2 = t0[:,1] + t0[:,3]/2; gy2 = t0[:,2] + t0[:,4]/2
                    gxyxy = np.stack([gx1, gy1, gx2, gy2], 1).astype(np.float32)
                    gcls  = t0[:,0].astype(np.int32)
                else:
                    gxyxy = np.zeros((0,4), np.float32); gcls = np.zeros((0,), np.int32)

                def iou_xyxy(a, b):
                    # a: [M,4], b: [N,4]
                    if a.size == 0 or b.size == 0:
                        return np.zeros((len(a), len(b)), dtype=np.float32)
                    xx1 = np.maximum(a[:,None,0], b[None,:,0])
                    yy1 = np.maximum(a[:,None,1], b[None,:,1])
                    xx2 = np.minimum(a[:,None,2], b[None,:,2])
                    yy2 = np.minimum(a[:,None,3], b[None,:,3])
                    iw = np.maximum(0.0, xx2-xx1)
                    ih = np.maximum(0.0, yy2-yy1)
                    inter = iw*ih
                    area_a = (a[:,2]-a[:,0])*(a[:,3]-a[:,1])
                    area_b = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])
                    union = area_a[:,None] + area_b[None,:] - inter
                    return inter / np.maximum(union, 1e-9)

                # Lower conf threshold to see if *any* match exists
                keep = (s0 >= 0.01)
                b0c, s0c, l0c = b0[keep], s0[keep], l0[keep]

                if len(gxyxy) and len(b0c):
                    ious = iou_xyxy(gxyxy, b0c)  # [Ng, Np]
                    best_pred_for_each_gt = ious.max(axis=1)          # [Ng]
                    best_score_for_each_gt = s0c[ious.argmax(axis=1)] # [Ng]
                    # Report the best IoU and its score
                    miou = float(best_pred_for_each_gt.max())
                    mscore = float(best_score_for_each_gt[best_pred_for_each_gt.argmax()])
                    print(f"[VAL-IoU-Probe] Ng={len(gxyxy)} Np(conf>=0.01)={len(b0c)} | best IoU={miou:.3f} @ score={mscore:.3f}")
                else:
                    print(f"[VAL-IoU-Probe] Ng={len(gxyxy)} Np(conf>=0.01)={len(b0c)} | no pairs to compare")

            if ib == 1:
                # class-agnostic IoU≥0.5 hits (first val batch only)
                def iou_xyxy(a, b):
                    if a.size == 0 or b.size == 0: return np.zeros((a.shape[0], b.shape[0]), np.float32)
                    x11,y11,x12,y12 = a[:,0:1],a[:,1:2],a[:,2:3],a[:,3:4]
                    x21,y21,x22,y22 = b[:,0],b[:,1],b[:,2],b[:,3]
                    iw = np.maximum(0, np.minimum(x12,x22)-np.maximum(x11,x21))
                    ih = np.maximum(0, np.minimum(y12,y22)-np.maximum(y11,y21))
                    inter = iw*ih
                    ua = np.maximum(0,(x12-x11)) * np.maximum(0,(y12-y11))
                    ub = np.maximum(0,(x22-x21)) * np.maximum(0,(y22-y21))
                    return inter / (ua + ub - inter + 1e-9)

                hits, tots = 0, 0
                for (pb, ps, pl), (gcls, gb) in zip(smoke_preds[:len(outputs)], smoke_gts[:len(outputs)]):
                    if len(gb) == 0: continue
                    tots += len(gb)
                    if len(pb) == 0: continue
                    ious = iou_xyxy(pb, gb)
                    hits += int((ious.max(axis=0) >= 0.5).sum())  # any pred overlaps each GT
                logger.info(f"[VAL-AGNOSTIC] IoU≥0.5 hits={hits}/{tots} on first batch")

                lab_list = [p[2] for p in smoke_preds[:len(outputs)] if len(p[2])]
                if len(lab_list):
                    all_labels = np.concatenate(lab_list, axis=0)
                else:
                    all_labels = np.zeros((0,), dtype=np.int32)
                if all_labels.size:
                    vals, cnts = np.unique(all_labels, return_counts=True)
                    mapping = ", ".join([f"{cfg['classes'][int(v)]}:{int(c)}" for v, c in zip(vals, cnts)])
                    logger.info(f"[VAL] pred class mix (first batch): {mapping}")  

            if ib == 1:
                pb, _, _ = outputs[0]
                gb = smoke_gts[0][1]
                pa = (np.maximum(0, pb[:,2]-pb[:,0]) * np.maximum(0, pb[:,3]-pb[:,1])).mean() if len(pb) else 0
                ga = (np.maximum(0, gb[:,2]-gb[:,0]) * np.maximum(0, gb[:,3]-gb[:,1])).mean() if len(gb) else 0
                logger.info(f"[VAL-AREA] mean pred area={pa:.1f}, mean gt area={ga:.1f}")   

            if ib == 1:
                # --- re-decode raw logits for image 0 (before threshold/NMS) ---
                raw = model_to_eval.forward(images)["p3"]  # [B, 5+C, Hs, Ws]
                B, K, Hs, Ws = raw.shape
                Himg, Wimg = images.shape[2], images.shape[3]
                sx = Wimg / float(Ws); sy = Himg / float(Hs)

                x = raw.permute(0, 2, 3, 1)  # [B,Hs,Ws,5+C]
                tx, ty, tw, th, tobj = torch.split(x[..., :5], 1, dim=-1)
                tcls = x[..., 5:]

                gy, gx = torch.meshgrid(
                    torch.arange(Hs, device=raw.device),
                    torch.arange(Ws, device=raw.device),
                    indexing="ij"
                )
                gx = gx[None, ..., None].float()
                gy = gy[None, ..., None].float()

                cx = (tx.sigmoid() + gx) * sx
                cy = (ty.sigmoid() + gy) * sy
                w  =  tw.exp() * sx
                h  =  th.exp() * sy

                # GT for image 0 in xyxy
                t0 = targets[0]
                if hasattr(t0, "cpu"): t0 = t0.cpu().numpy()
                if t0.size:
                    gx1 = t0[:,1] - t0[:,3]/2; gy1 = t0[:,2] - t0[:,4]/2
                    gx2 = t0[:,1] + t0[:,3]/2; gy2 = t0[:,2] + t0[:,4]/2
                    gxyxy = np.stack([gx1, gy1, gx2, gy2], 1).astype(np.float32)
                else:
                    gxyxy = np.zeros((0,4), np.float32)

                # flatten all cells for image 0
                cx0 = cx[0, ..., 0].reshape(-1)
                cy0 = cy[0, ..., 0].reshape(-1)
                w0  =  w[0,  ..., 0].reshape(-1)
                h0  =  h[0,  ..., 0].reshape(-1)
                x1 = (cx0 - w0/2).clamp(0, Wimg-1)
                y1 = (cy0 - h0/2).clamp(0, Himg-1)
                x2 = (cx0 + w0/2).clamp(0, Wimg-1)
                y2 = (cy0 + h0/2).clamp(0, Himg-1)
                boxes0 = torch.stack([x1,y1,x2,y2], dim=1).cpu().numpy().astype(np.float32)  # [Hs*Ws,4]

                def _iou(a, b):
                    if len(a)==0 or len(b)==0: return np.zeros((len(a), len(b)), np.float32)
                    xx1 = np.maximum(a[:,None,0], b[None,:,0])
                    yy1 = np.maximum(a[:,None,1], b[None,:,1])
                    xx2 = np.minimum(a[:,None,2], b[None,:,2])
                    yy2 = np.minimum(a[:,None,3], b[None,:,3])
                    iw = np.maximum(0.0, xx2-xx1); ih = np.maximum(0.0, yy2-yy1)
                    inter = iw*ih
                    area_a = (a[:,2]-a[:,0])*(a[:,3]-a[:,1])
                    area_b = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])
                    union = area_a[:,None] + area_b[None,:] - inter
                    return inter / np.maximum(union, 1e-9)

                if len(gxyxy):
                    ious_all = _iou(gxyxy, boxes0)     # [Ng, Hs*Ws]
                    best_iou_all = float(ious_all.max())
                    # also check the *assigned* cell (where the GT center falls)
                    gt = t0[0]  # take first GT
                    ix = int(gt[1] / sx); iy = int(gt[2] / sy)
                    ix = max(0, min(Ws-1, ix)); iy = max(0, min(Hs-1, iy))
                    k  = iy*Ws + ix
                    best_iou_cell = float(ious_all[:, k].max())
                    print(f"[VAL-CELL-PROBE] Hs,Ws=({Hs},{Ws}) stride=({sx:.1f},{sy:.1f}) | IoU_all_max={best_iou_all:.3f} | IoU_at_cell({iy},{ix})={best_iou_cell:.3f}")
                else:
                    print("[VAL-CELL-PROBE] no GT in first image")   

            if ib == 1 and len(gxyxy):
                # Compute scores per cell BEFORE threshold/NMS for image 0
                # reuse raw, tcls, tobj, Hs, Ws, sx, sy from the block above
                obj_prob = tobj[0, ..., 0].sigmoid()           # [Hs,Ws]
                cls_prob = tcls[0].sigmoid()                   # [Hs,Ws,C]
                max_cls_prob, max_cls_id = cls_prob.max(dim=-1)  # [Hs,Ws]

                # Find the cell with MAX IoU to the GT we just measured
                k_best = int(np.argmax(ious_all[0]))           # flat index among Hs*Ws
                iy_best, ix_best = divmod(k_best, Ws)

                # Score at the best-IoU cell
                score_best = float((obj_prob[iy_best, ix_best] * max_cls_prob[iy_best, ix_best]).item())
                best_lab  = int(max_cls_id[iy_best, ix_best].item())

                # Score at the ASSIGNED cell (iy, ix) from the earlier calc
                score_cell = float((obj_prob[iy, ix] * max_cls_prob[iy, ix]).item())
                cell_lab   = int(max_cls_id[iy, ix].item())

                print(f"[VAL-SCORE-PROBE] IoU_all_max={best_iou_all:.3f} at cell({iy_best},{ix_best}) "
                    f"| score_best={score_best:.4f} lab_best={best_lab} "
                    f"| score_at_assigned({iy},{ix})={score_cell:.4f} lab_cell={cell_lab}")  

            if ib == 1:
                b0, s0, l0 = outputs[0]
                if len(b0) and len(gxyxy):
                    # best IoU among final outputs
                    ious_out = iou_xyxy(gxyxy, b0)        # reuse your IoU helper
                    m = ious_out[0].argmax()
                    print(f"[VAL-OUT-PROBE] best IoU from outputs = {float(ious_out[0,m]):.3f}, score={float(s0[m]):.3f}, label={int(l0[m])}")  

            if ib == 1 and len(gxyxy):
                # 1) Rebuild the assigned-cell box (same decode you used above)
                # we already have: raw, Hs, Ws, sx, sy, tx, ty, tw, th
                cx_cell = (tx[0, iy, ix, 0].sigmoid() + ix) * sx
                cy_cell = (ty[0, iy, ix, 0].sigmoid() + iy) * sy
                w_cell  =  tw[0, iy, ix, 0].exp() * sx
                h_cell  =  th[0, iy, ix, 0].exp() * sy
                ac_x1 = float(max(0, cx_cell - w_cell/2))
                ac_y1 = float(max(0, cy_cell - h_cell/2))
                ac_x2 = float(min(Wimg-1, cx_cell + w_cell/2))
                ac_y2 = float(min(Himg-1, cy_cell + h_cell/2))

                # 2) Compare this assigned-cell box to the postprocess outputs
                b0, s0, l0 = outputs[0]  # numpy arrays
                if len(b0):
                    # IoU between assigned cell box and every output
                    ox1 = b0[:,0]; oy1 = b0[:,1]; ox2 = b0[:,2]; oy2 = b0[:,3]
                    xx1 = np.maximum(ox1, ac_x1); yy1 = np.maximum(oy1, ac_y1)
                    xx2 = np.minimum(ox2, ac_x2); yy2 = np.minimum(oy2, ac_y2)
                    iw = np.maximum(0.0, xx2-xx1); ih = np.maximum(0.0, yy2-yy1)
                    inter = iw*ih
                    area_o = (ox2-ox1)*(oy2-oy1)
                    area_a = (ac_x2-ac_x1)*(ac_y2-ac_y1)
                    iou_to_assigned = inter / np.maximum(area_o + area_a - inter, 1e-9)

                    j = int(iou_to_assigned.argmax())
                    print(f"[VAL-ASSIGNED-IN-OUT] max IoU between assigned-cell box and outputs = {float(iou_to_assigned[j]):.3f}, "
                        f"score_out={float(s0[j]):.3f}, label_out={int(l0[j])}")
                else:
                    print("[VAL-ASSIGNED-IN-OUT] no outputs to compare")
    # ---- compute ORIGINAL metrics ----
    metrics = ap50_eval.compute()
    ap50 = metrics.get("ap50", 0.0)
    logger.info(f"[VAL] AP50={ap50:.4f}")
    if cfg["metrics"].get("per_class", True):
        pcs = metrics.get("per_class", {})
        for cls_name, pr in pcs.items():
            logger.info(f"  {cls_name:<20} P={pr['precision']:.3f} R={pr['recall']:.3f} AP50={pr['ap50']:.3f}")

    # ---- compute SMOKE AP50 ----
    tot_preds = sum(len(p[1]) for p in smoke_preds)  # p=(boxes,scores,labels)
    tot_gts   = sum(len(g[0]) for g in smoke_gts)    # g=(cls_ids, boxes)
    logger.info(f"[VAL-SMOKE] sanity: imgs={len(smoke_preds)} "
                f"tot_preds={tot_preds} tot_gts={tot_gts} "
                f"mean_preds/img={tot_preds/max(1,len(smoke_preds)):.1f} "
                f"mean_gts/img={tot_gts/max(1,len(smoke_gts)):.1f}")

    mAP50, ap_per_cls, P_cls, R_cls = ap50_from_batches(
        batched_preds=smoke_preds,
        batched_gts=smoke_gts,
        num_classes=cfg["num_classes"],
        iou_th=0.5,
    )
    logger.info(f"[VAL-SMOKE] mAP50={mAP50:.4f}")
    for i, name in enumerate(cfg["classes"]):
        logger.info(f"  {name:<18} P={P_cls[i]:.3f} R={R_cls[i]:.3f} AP50={ap_per_cls[i]:.3f}")

    return metrics


def train(cfg_path: str):
    # ---- Setup
    cfg = load_config(cfg_path)
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(save_dir / "train.log")
    logger.info(f"Loaded config from {cfg_path}")
    logger.info(f"Save dir: {save_dir}")

    set_seed(int(cfg.get("seed", 0)))
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    use_amp = bool(cfg.get("amp", True))
    scaler = torch_amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    # ---- Data
    train_set, val_set, train_loader, val_loader = build_dataloaders(cfg, device)
    logger.info(f"Train images: {len(train_set)} | Val images: {len(val_set)}")
    # ---- Model
    model = build_model(cfg).to(device)
    n_params = count_params(model)
    logger.info(f"Model params: {n_params:,} ({human_bytes(n_params*4)})")
    # ---- Loss & Assigner
    assigner = NaiveGridAssigner(strides=cfg["strides"], pos_cells=cfg.get("pos_cells", 1))
    criterion = YoloLoss(cfg, assigner, cfg.get("num_classes")).to(device)
    # ---- Optim / Sched
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(optimizer, cfg, iters_per_epoch=len(train_loader))
    # ---- AMP / EMA
    use_amp = bool(cfg.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    use_ema = bool(cfg.get("ema", True))
    ema = ModelEMA(model) if use_ema else None

    # ---- Resume
    start_epoch = resume_if_possible(logger, model, optimizer, scaler, cfg, ema)

    # ---- Best tracking
    best_metric_name = cfg.get("best_metric", cfg.get("save_best_metric", "ap50"))
    best_mode = cfg.get("best_mode", "max")
    best_value = -1e9 if best_mode == "max" else 1e9
    save_best = bool(cfg.get("save_best", True))

    # ---- Training loop
    epochs = int(cfg["epochs"])
    grad_accum = int(cfg.get("grad_accum", 1))
    eval_interval = int(cfg.get("eval_interval", 5))
    model.train()

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Progress bar
        bar = TqdmBar(total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")

        running = {"loss": 0.0, "l_box": 0.0, "l_obj": 0.0, "l_cls": 0.0}
        for it, (images, targets) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            targets = [t.to(device) for t in targets]

            with torch.amp.autocast('cuda', enabled=(scaler is not None and device.type == "cuda")):
                # forward: model should return a dict of raw predictions per scale
                preds = model(images)
                loss, parts = criterion(preds, targets, img_size=cfg["img_size"])

            # backward
            loss = loss / grad_accum
            scaler.scale(loss).backward()

            if it % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

                # EMA update
                if ema is not None:
                    ema.update(model)

            # logging
            running["loss"] += float(loss) * grad_accum
            running["l_box"] += float(parts.get("l_box", 0.0))
            running["l_obj"] += float(parts.get("l_obj", 0.0))
            running["l_cls"] += float(parts.get("l_cls", 0.0))
            bar.set_postfix(
                loss=f"{running['loss']/it:.4f}",
                l_box=f"{running['l_box']/it:.3f}",
                l_obj=f"{running['l_obj']/it:.3f}",
                l_cls=f"{running['l_cls']/it:.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}" if scheduler else f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            bar.update(1)
        bar.close()
        dt = time.time() - t0
        logger.info(f"Epoch {epoch} finished in {dt:.1f}s | loss={running['loss']/max(1,len(train_loader)):.4f}")

        # ---- Save last.pt
        if bool(cfg.get("save_last", True)):
            save_ckpt(save_dir / "last.pt", model, optimizer, epoch, scaler, cfg, ema=ema)

        # ---- Eval & best.pt
        if eval_interval > 0 and (epoch % eval_interval == 0 or epoch == epochs):
            metrics = validate(model, val_loader, device, cfg, logger, ema=ema)
            current = float(metrics.get(best_metric_name, 0.0))
            improved = (current > best_value + 1e-6) if best_mode == "max" else (current < best_value - 1e-6)
            if save_best and improved:
                best_value = current
                save_ckpt(save_dir / "best.pt", model, optimizer, epoch, scaler, cfg, ema=ema, metric_value=current)
                logger.info(f"Saved best.pt (epoch {epoch}, {best_metric_name}={current:.4f})")

    logger.info("Training complete.")


def parse_args():
    ap = argparse.ArgumentParser(description="YOLO training based on the yolox model")
    ap.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to YAML config (custom_yolo/configs/yolo_train.yaml)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)