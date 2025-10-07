from __future__ import annotations
import torch

class NaiveGridAssigner:
    """
    Minimal assigner: for each GT, pick the nearest grid cell on the (only) stride.
    Optionally mark a small neighborhood (pos_cells) around the center as positives.
    """
    def __init__(self, strides, pos_cells: int = 1):
        assert isinstance(strides, (list, tuple)) and len(strides) == 1, \
            "NaiveGridAssigner currently supports single-scale (one stride)."
        self.stride = int(strides[0])
        self.pos_cells = int(pos_cells)

    def __call__(self, targets_list, grid_hw, img_hw):
        """
        Args:
          targets_list: list[Tensor(N_i, 5)] each row = (cls, cx, cy, w, h) in pixels on network canvas
          feat_hw: (Hs, Ws)
        Returns:
          list of dicts with per-image target maps:
            {
              "obj":  [Hs, Ws, 1] 0/1
              "box":  [Hs, Ws, 4] (cx,cy,w,h) targets (zeros elsewhere)
              "cls":  [Hs, Ws, C] one-hot on positives
              "mask": [Hs, Ws, 1] 1 for positive cells (helps masking losses)
            }
        """
        Hs, Ws = grid_hw
        Himg, Wimg = img_hw
        sx = Wimg / float(Ws)
        sy = Himg / float(Hs)

        out = []
        for t in targets_list:
            # t: [N,5] [cls, cx, cy, w, h] in PIXELS (your dataset loader already multiplies by img_size)
            obj = torch.zeros((Hs, Ws, 1), device=t.device)
            box = torch.zeros((Hs, Ws, 4), device=t.device)
            mask = torch.zeros((Hs, Ws, 1), device=t.device)

            if t.numel() > 0:
                cx, cy, w, h = t[:,1], t[:,2], t[:,3], t[:,4]
                # which cell?
                ix = (cx / sx).long().clamp_(0, Ws-1)
                iy = (cy / sy).long().clamp_(0, Hs-1)

                r = getattr(self, "pos_cells", 1) // 2  # if pos_cells=3 -> r=1; if 1 -> r=0
                # ensure odd window; if pos_cells is even, make it odd
                if getattr(self, "pos_cells", 1) % 2 == 0:
                    r = max(0, (self.pos_cells - 1) // 2)

                for k in range(len(cx)):
                    cxk, cyk, wk, hk = cx[k], cy[k], w[k], h[k]
                    ixc = int(ix[k].item()); iyc = int(iy[k].item())
                    x1 = max(0, ixc - r); x2 = min(Ws - 1, ixc + r)
                    y1 = max(0, iyc - r); y2 = min(Hs - 1, iyc + r)

                    obj[y1:y2+1, x1:x2+1, 0]  = 1.0
                    mask[y1:y2+1, x1:x2+1, 0] = 1.0
                    # put the same GT box target in the neighborhood (simple & effective)
                    box[y1:y2+1, x1:x2+1, 0] = cxk
                    box[y1:y2+1, x1:x2+1, 1] = cyk
                    box[y1:y2+1, x1:x2+1, 2] = wk
                    box[y1:y2+1, x1:x2+1, 3] = hk

                obj[iy, ix, 0]  = 1.0
                box[iy, ix, :]  = torch.stack([cx, cy, w, h], dim=1)  # keep in PIXELS
                mask[iy, ix, 0] = 1.0

                # (optional) expand to neighbors if you use pos_cells>1

            out.append({"obj": obj, "box": box, "mask": mask})
        return out