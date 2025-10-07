from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import Dataset
from .transforms import build_transforms

def _read_txt(p: Path):
    if not p.exists(): return np.zeros((0, 5), dtype=np.float32)
    anns = []
    with p.open() as f:
        for ln in f:
            sp = ln.strip().split()
            if len(sp)!=5: continue
            c, cx, cy, w, h = map(float, sp)
            anns.append([int(c), cx, cy, w, h])
    return np.array(anns, dtype=np.float32) if len(anns) else np.zeros((0,5), np.float32)

class YoloTxtDataset(Dataset):
    def __init__(self, root, img_subdir, lbl_subdir, classes, img_size=640, transforms=None, split="train"):
        self.root = Path(root)
        self.img_dir = (self.root / img_subdir)
        self.lbl_dir = (self.root / lbl_subdir)
        self.names = classes
        self.size = int(img_size)
        self.split = split
        self.paths = sorted([p for p in self.img_dir.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}])
        assert len(self.paths)>0, f"No images found in {self.img_dir}"
        self.transforms = transforms
        if self.transforms is None:
            # fallback if called directly
            from .transforms import build_transforms as _bt
            self.transforms = _bt({"img_size": self.size})[0 if split=="train" else 1]

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        ip = self.paths[i]
        img = cv2.imread(str(ip))
        if img is None: raise FileNotFoundError(f"Failed to read {ip}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lp = self.lbl_dir / (ip.stem + ".txt")
        t = _read_txt(lp)  # (N,5) class,cx,cy,w,h in 0..1 relative to original

        # apply transforms (handles letterbox + aug + to tensor)
        if callable(self.transforms):
            img_t, t = self.transforms(img, t.copy())
        else:
            img_t, t = img, t

        if not isinstance(t, np.ndarray):
            t = np.array(t, dtype=np.float32).reshape(-1, 5) if len(t) else np.zeros((0,5), np.float32)

        # return Tensor image and Tensor targets (float32)
        return img_t, torch.from_numpy(t)
