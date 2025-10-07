import random, cv2, numpy as np
import torch

def letterbox(img, new=640, color=(114,114,114)):
    h, w = img.shape[:2]
    s = min(new/h, new/w)
    nh, nw = int(round(h*s)), int(round(w*s))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new - nh)//2; bottom = new - nh - top
    left = (new - nw)//2; right = new - nw - left
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, s, left, top, w, h

class TrainTransforms:
    def __init__(self, size=640, hflip=0.5, brightness=0.2, contrast=0.2,
                 rotation_deg=8, motion_blur_p=0.2, random_rescale=(0.8,1.2)):
        self.size = int(size)
        self.hflip = float(hflip)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.rotation_deg = float(rotation_deg)
        self.motion_blur_p = float(motion_blur_p)
        self.random_rescale = tuple(random_rescale) if random_rescale else (1.0, 1.0)

    def __call__(self, img, targets):
        # letterbox
        img, s, px, py, w0, h0 = letterbox(img, self.size)
        if len(targets):
            targets[:,1] = targets[:,1]*w0*s + px
            targets[:,2] = targets[:,2]*h0*s + py
            targets[:,3] = targets[:,3]*w0*s
            targets[:,4] = targets[:,4]*h0*s

        H, W = img.shape[:2]
        # hflip
        if random.random() < self.hflip:
            img = img[:, ::-1].copy()
            if len(targets):
                targets[:,1] = W - targets[:,1]

        # brightness/contrast
        if random.random() < 0.5:
            a = 1.0 + self.contrast*(random.random()*2-1)
            b = 255.0*self.brightness*(random.random()*2-1)
            img = np.clip(img.astype(np.float32)*a + b, 0, 255).astype(np.uint8)

        # (optional) motion blur
        if random.random() < self.motion_blur_p:
            k = random.choice([3,5])
            img = cv2.GaussianBlur(img, (k,k), 0)

        img = torch.from_numpy(img.astype(np.float32)/255.0).permute(2,0,1)
        return img, targets

class ValTransforms:
    def __init__(self, size=640, debug_once=False):
        self.size = int(size)
        self.debug_once = bool(debug_once)
        self._did_debug = False

    def __call__(self, img, targets):
        img, s, px, py, w0, h0 = letterbox(img, self.size)
        if len(targets):
            targets[:,1] = targets[:,1]*w0*s + px
            targets[:,2] = targets[:,2]*h0*s + py
            targets[:,3] = targets[:,3]*w0*s
            targets[:,4] = targets[:,4]*h0*s

        if self.debug_once and not self._did_debug:
            self._did_debug = True
            H, W = img.shape[:2]
            n = int(len(targets))
            if n:
                t = targets.copy()  # numpy or torch.Tensor both fine for .copy()/.cpu().numpy() if needed
                # If it's a tensor, uncomment the next line:
                # t = t.cpu().numpy()
                min_xy = (float(t[:,1].min()), float(t[:,2].min()))
                max_xy = (float(t[:,1].max()), float(t[:,2].max()))
                max_wh = (float(t[:,3].max()), float(t[:,4].max()))
            else:
                min_xy = max_xy = max_wh = None

            print(f"[VAL-SANITY] orig_wh=({w0},{h0}) scale={s:.4f} pad=({px},{py}) "
                  f"img_hw=({H},{W}) n_targets={n} min_xy={min_xy} max_xy={max_xy} max_wh={max_wh}")

            # Also dump a quick visualization of GT boxes *after* transform
            try:
                vis = img.copy()
                if n:
                    for x in t:
                        cx, cy, w, h = x[1], x[2], x[3], x[4]
                        x1 = int(cx - w/2); y1 = int(cy - h/2)
                        x2 = int(cx + w/2); y2 = int(cy + h/2)
                        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.imwrite("/mnt/data/_val_sanity_0.jpg", vis)
                print("[VAL-SANITY] wrote /mnt/data/_val_sanity_0.jpg")
            except Exception as e:
                print(f"[VAL-SANITY] viz failed: {e}")
        
        img = torch.from_numpy(img.astype(np.float32)/255.0).permute(2,0,1)
        return img, targets

def build_transforms(cfg):
    size = cfg["img_size"]
    train_tfms = TrainTransforms(
        size=size,
        hflip=cfg.get("hflip", 0.5),
        brightness=cfg.get("brightness", 0.2),
        contrast=cfg.get("contrast", 0.2),
        rotation_deg=cfg.get("rotation_deg", 8),
        motion_blur_p=cfg.get("motion_blur_p", 0.2),
        random_rescale=cfg.get("random_rescale", [1.0, 1.0]),
    )
    val_tfms = ValTransforms(size=size, debug_once=True)
    return train_tfms, val_tfms