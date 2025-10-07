import os
from pathlib import Path
import argparse
import cv2
import numpy as np
import torch

# local imports
from yolo.utils.config import load_config
from yolo.data.dataset_yolo_txt import YoloTxtDataset
from yolo.data.transforms import build_transforms

def draw_boxes(img_rgb, targets, color=(0,255,0)):
    """targets: ndarray[N,5] with (cls, cx, cy, w, h) in **pixels on the network canvas**"""
    out = img_rgb.copy()
    H, W = out.shape[:2]
    bad = 0
    for (c, cx, cy, w, h) in targets:
        x1 = int(round(cx - w/2)); y1 = int(round(cy - h/2))
        x2 = int(round(cx + w/2)); y2 = int(round(cy + h/2))
        if x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H or w <= 0 or h <= 0:
            bad += 1
            clr = (0,0,255)  # red for out-of-frame/degenerate
        else:
            clr = color
        cv2.rectangle(out, (x1, y1), (x2, y2), clr, 2)
        cv2.putText(out, str(int(c)), (max(0,x1), max(10,y1-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1, cv2.LINE_AA)
    return out, bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to YAML config")
    ap.add_argument("--split", default="train", choices=["train","val"])
    ap.add_argument("--num", type=int, default=32, help="How many samples to dump")
    ap.add_argument("--out", default="runs/debug_vis", help="Output dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    size = int(cfg["img_size"])

    # build transforms (same as training)
    train_tfms, val_tfms = build_transforms(cfg)
    tfms = train_tfms if args.split == "train" else val_tfms

    img_subdir = cfg["train_images"] if args.split=="train" else cfg["val_images"]
    lbl_subdir = cfg["train_labels"] if args.split=="train" else cfg["val_labels"]

    ds = YoloTxtDataset(
        root=cfg["data_root"],
        img_subdir=img_subdir,
        lbl_subdir=lbl_subdir,
        classes=cfg["classes"],
        img_size=size,
        transforms=tfms,
        split=args.split,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(args.num, len(ds))
    print(f"Dumping {n} {args.split} samples to {out_dir} (canvas {size}x{size})")

    total_bad = 0
    for i in range(n):
        img_t, t = ds[i]  # img_t: torch CHW (0..1), t: Tensor[N,5] in PX on canvas
        img = (img_t.permute(1,2,0).numpy() * 255).astype(np.uint8).copy()
        t_np = t.numpy() if isinstance(t, torch.Tensor) else t

        vis, bad = draw_boxes(img, t_np, (0,255,0))
        total_bad += bad

        # grid overlay (stride=8 by default) to sanity check cell indexing
        stride = int(cfg["strides"][0])
        if stride > 0:
            for x in range(0, size, stride):
                cv2.line(vis, (x,0), (x,size), (60,60,60), 1)
            for y in range(0, size, stride):
                cv2.line(vis, (0,y), (size,y), (60,60,60), 1)

        # save
        out_path = out_dir / f"{args.split}_{i:04d}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        # quick stats line
        if i % 8 == 0:
            print(f"[{i:4d}] boxes={len(t_np)} bad={bad}")

    print(f"Done. Total bad/degenerate or off-canvas boxes across {n} samples: {total_bad}")

if __name__ == "__main__":
    main()