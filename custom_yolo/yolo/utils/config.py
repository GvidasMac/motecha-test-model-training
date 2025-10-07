from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # PyYAML
except ImportError as e:
    raise ImportError("PyYAML not found. Install with: pip install pyyaml") from e


# -------- Defaults (applied if missing in YAML) --------
_DEFAULTS: Dict[str, Any] = {
    # dataset
    "train_images": "images/train",
    "train_labels": "labels/train",
    "val_images": "images/val",
    "val_labels": "labels/val",
    "img_size": 640,

    # model
    "backbone": "mobilenetv3_small",
    "neck": None,
    "strides": [8],
    "act": "silu",
    "head_channels": 64,

    # training
    "epochs": 150,
    "batch_size": 16,
    "grad_accum": 1,
    "num_workers": 4,
    "optimizer": "adamw",
    "lr": 3e-4,
    "weight_decay": 5e-4,
    "ema": True,
    "amp": True,
    "warmup_epochs": 3,
    "scheduler": "cosine",
    "resume": True,
    "save_last": True,

    # augmentation
    "hflip": 0.5,
    "brightness": 0.2,
    "contrast": 0.2,
    "rotation_deg": 8,
    "motion_blur_p": 0.2,
    "hsv_jitter": 0.0,
    "random_rescale": [0.8, 1.2],
    "mosaic": False,
    "mixup": False,

    # loss & assign
    "box_loss": "ciou",
    "cls_loss": "bce",
    "obj_loss": "bce",
    "assigner": "naive_grid",
    "pos_cells": 1,
    "obj_pos_weight": 1.0,
    "cls_pos_weight": 1.0,
    "box_weight": 1.0,

    # metrics
    "metrics": {
        "ap50": True,
        "per_class": True,
        "latency_probe": True,
        "iou_thresholds": [0.5],
    },

    # validation / ckpt
    "eval_interval": 5,
    "save_dir": "runs/exp",
    "save_best": True,
    "best_metric": "ap50",
    "best_mode": "max",
    "ckpt_keep_last": 2,

    # misc
    "seed": 0,
    "save_batch_preview": False,
    "preview_dir": "runs/exp/preview",
}


def _merge_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge user YAML over defaults."""
    def merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(a)
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = v
        return out
    return merge(_DEFAULTS, cfg)


def _req(cfg: Dict[str, Any], key: str):
    if key not in cfg or cfg[key] is None or cfg[key] == "":
        raise ValueError(f"Missing required config key: '{key}'")


def _check_paths(cfg: Dict[str, Any]) -> None:
    root = Path(cfg["data_root"]).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {root}")

    # Expand relative subpaths against data_root
    for k in ["train_images", "train_labels", "val_images", "val_labels"]:
        p = root / cfg[k]
        if not p.exists():
            raise FileNotFoundError(f"{k} path not found: {p}")
        cfg[k] = str(p)  # store absolute for convenience

    # Ensure save dirs exist
    Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg.get("preview_dir", str(Path(cfg["save_dir"]) / "preview"))).mkdir(parents=True, exist_ok=True)


def _validate_classes(cfg: Dict[str, Any]) -> None:
    classes = cfg.get("classes", None)
    _req(cfg, "classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("classes must be a non-empty list of class names.")
    if "num_classes" in cfg:
        if int(cfg["num_classes"]) != len(classes):
            raise ValueError(
                f"num_classes ({cfg['num_classes']}) does not match len(classes) ({len(classes)}). "
                "Remove num_classes or make them consistent."
            )
    cfg["num_classes"] = len(classes)


def _validate_strides(cfg: Dict[str, Any]) -> None:
    strides = cfg.get("strides", [8])
    if not isinstance(strides, list) or not all(isinstance(s, int) for s in strides):
        raise ValueError("strides must be a list of integers, e.g., [8] or [8,16,32].")
    if cfg["img_size"] % min(strides) != 0:
        raise ValueError(f"img_size ({cfg['img_size']}) must be divisible by smallest stride ({min(strides)}).")


def _normalize_types(cfg: Dict[str, Any]) -> None:
    # Ensure some numeric fields are int/float
    cfg["img_size"] = int(cfg["img_size"])
    cfg["epochs"] = int(cfg["epochs"])
    cfg["batch_size"] = int(cfg["batch_size"])
    cfg["grad_accum"] = int(cfg.get("grad_accum", 1))
    cfg["num_workers"] = int(cfg.get("num_workers", 4))
    cfg["lr"] = float(cfg["lr"])
    cfg["weight_decay"] = float(cfg["weight_decay"])
    cfg["warmup_epochs"] = int(cfg["warmup_epochs"])
    cfg["eval_interval"] = int(cfg["eval_interval"])
    cfg["seed"] = int(cfg.get("seed", 0))
    # Booleans already safe via YAML, but ensure presence
    for k in ["ema", "amp", "resume", "save_last", "save_best", "mosaic", "mixup"]:
        cfg[k] = bool(cfg.get(k, _DEFAULTS[k]))


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """
    Load YAML, merge defaults, validate, and normalize.
    Returns a plain dict used by train.py and the rest of the stack.
    """
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}

    if not isinstance(user_cfg, dict):
        raise ValueError("Top-level YAML must be a mapping/dictionary.")

    # Merge with defaults
    cfg = _merge_defaults(user_cfg)

    # Required keys
    _req(cfg, "data_root")
    _req(cfg, "classes")

    # Validate content
    _check_paths(cfg)
    _validate_classes(cfg)
    _validate_strides(cfg)
    _normalize_types(cfg)

    # Echo some useful derived values
    cfg["class_to_id"] = {name: i for i, name in enumerate(cfg["classes"])}
    return cfg