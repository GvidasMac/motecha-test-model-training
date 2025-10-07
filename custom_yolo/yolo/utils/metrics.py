import numpy as np

def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    w = max(0, min(ax2, bx2) - max(ax1, bx1))
    h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = w*h
    ua = max(0, ax2-ax1) * max(0, ay2-ay1) + max(0, bx2-bx1) * max(0, by2-by1) - inter + 1e-6
    return inter / ua

class EvalAP50:
    def __init__(self, class_names, iou_thresh=0.5):
        self.names = class_names
        self.iou = float(iou_thresh)
        self.tps = [0]*len(class_names)
        self.fps = [0]*len(class_names)
        self.fns = [0]*len(class_names)

    def add_batch(self, outputs, targets_list, img_size):
        # outputs: list of (boxes_xyxy np[N,4], scores np[N], labels np[N])
        # targets_list: list of Tensor[N_i, 5] with (cls,cx,cy,w,h) in pixels on the network image
        for (boxes, scores, labels), t in zip(outputs, targets_list):
            t = t.cpu().numpy() if hasattr(t, "cpu") else t
            gt_boxes = []
            gt_cls = []
            for r in t:
                c, cx, cy, w, h = r
                gt_boxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
                gt_cls.append(int(c))
            matched = set()
            for b, s, l in zip(boxes, scores, labels):
                if len(gt_boxes)==0:
                    self.fps[l]+=1; continue
                ious = np.array([_iou_xyxy(b, gtb) for gtb in gt_boxes])
                j = int(ious.argmax())
                if ious[j] >= self.iou and l == gt_cls[j] and j not in matched:
                    self.tps[l]+=1; matched.add(j)
                else:
                    self.fps[l]+=1
            for j in range(len(gt_boxes)):
                if j not in matched:
                    self.fns[gt_cls[j]] += 1

    def compute(self):
        per = {}
        ap50_sum = 0; n_with_gt = 0
        for i, name in enumerate(self.names):
            tp, fp, fn = self.tps[i], self.fps[i], self.fns[i]
            prec = tp / max(1, tp + fp)
            rec  = tp / max(1, tp + fn)
            # crude AP50 proxy = P*R at single threshold (good enough to track progress)
            ap50 = prec * rec
            per[name] = {"precision": prec, "recall": rec, "ap50": ap50, "tp": tp, "fp": fp, "fn": fn}
            if tp+fn > 0:
                ap50_sum += ap50; n_with_gt += 1
        overall = ap50_sum / max(1, n_with_gt)
        return {"ap50": overall, "per_class": per}