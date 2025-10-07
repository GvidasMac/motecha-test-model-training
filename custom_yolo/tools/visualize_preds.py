import argparse, torch, cv2, numpy as np
from pathlib import Path
from yolo.utils.config import load_config
from yolo.data.dataset_yolo_txt import YoloTxtDataset
from yolo.data.transforms import build_transforms
from yolo.models import build_model

def draw_xyxy(img, box, color, txt=None):
    x1,y1,x2,y2 = [int(round(v)) for v in box]
    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
    if txt:
        cv2.putText(img, txt, (x1, max(10,y1-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", required=True)
    ap.add_argument("--ckpt", required=True, help="path to last.pt/best.pt")
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--out", default="runs/debug_preds")
    ap.add_argument("--conf", type=float, default=0.10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    _, val_tfms = build_transforms(cfg)

    ds = YoloTxtDataset(cfg["data_root"], cfg["val_images"], cfg["val_labels"],
                        cfg["classes"], img_size=cfg["img_size"], transforms=val_tfms, split="val")

    # build model and load weights
    model = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(args.num, len(ds))):
        img_t, t = ds[i]
        img = (img_t.permute(1,2,0).numpy()*255).astype(np.uint8).copy()

        with torch.no_grad():
            outs = model.postprocess(img_t[None], conf_thres=args.conf, nms_thres=0.7)[0]
        boxes, scores, labels = outs

        # draw GT (green)
        for (c, cx, cy, w, h) in t.numpy():
            x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
            draw_xyxy(img, (x1,y1,x2,y2), (0,255,0), f"gt:{int(c)}")

        # draw predictions (blue)
        for b, s, l in zip(boxes, scores, labels):
            draw_xyxy(img, b, (255,128,0), f"{cfg['classes'][int(l)]}:{s:.2f}")

        cv2.imwrite(str(out_dir / f"pred_{i:03d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()