import numpy as np

def box_iou_xyxy(a, b):
    # a: [Na,4], b: [Nb,4]
    an = a.shape[0]; bn = b.shape[0]
    if an == 0 or bn == 0:
        return np.zeros((an, bn), dtype=np.float32)
    x11, y11, x12, y12 = a[:,0,None], a[:,1,None], a[:,2,None], a[:,3,None]
    x21, y21, x22, y22 = b[:,0],      b[:,1],      b[:,2],      b[:,3]
    iw = np.maximum(0, np.minimum(x12, x22) - np.maximum(x11, x21))
    ih = np.maximum(0, np.minimum(y12, y22) - np.maximum(y11, y21))
    inter = iw * ih
    area_a = np.maximum(0, (x12 - x11)) * np.maximum(0, (y12 - y11))
    area_b = np.maximum(0, (x22 - x21)) * np.maximum(0, (y22 - y21))
    union = area_a + area_b - inter + 1e-9
    return inter / union

def ap50_from_batches(batched_preds, batched_gts, num_classes, iou_th=0.5):
    """
    batched_preds: list over images of (boxes[N,4] xyxy, scores[N], labels[N])
    batched_gts  : list over images of (cls_ids[M], boxes_xyxy[M,4])
    returns: (mAP50, per_class_ap50, per_class_P, per_class_R)
    """
    C = num_classes
    aps = np.zeros(C, dtype=np.float32)
    prec = np.zeros(C, dtype=np.float32)
    rec  = np.zeros(C, dtype=np.float32)

    for c in range(C):
        # collect all predictions of class c across images
        P_boxes, P_scores, G_boxes = [], [], []
        G_used_flags = []  # per image arrays

        for (pred, gt) in zip(batched_preds, batched_gts):
            pb, ps, pl = pred
            m = (pl == c)
            P_boxes.append(pb[m])
            P_scores.append(ps[m])

            g_cls, g_boxes = gt
            mgt = (g_cls == c)
            G_boxes.append(g_boxes[mgt])
            G_used_flags.append(np.zeros((g_boxes[mgt].shape[0],), dtype=bool))

        P_boxes = np.concatenate(P_boxes, axis=0) if len(P_boxes) else np.zeros((0,4), np.float32)
        P_scores = np.concatenate(P_scores, axis=0) if len(P_scores) else np.zeros((0,), np.float32)

        # no GT of this class
        total_gts = int(sum([len(g) for g in G_boxes]))
        if total_gts == 0:
            aps[c] = 0.0; prec[c]=0.0; rec[c]=0.0
            continue

        # sort preds by confidence
        order = np.argsort(-P_scores)
        P_boxes = P_boxes[order]
        P_scores = P_scores[order]

        tp = np.zeros(len(P_scores), dtype=np.float32)
        fp = np.zeros(len(P_scores), dtype=np.float32)

        # match predictions to GTs per image
        # we need image-wise grouping again to check GT usage; simpler approach:
        # rewalk images and greedily match per image
        idx = 0
        for (pred, gt, used) in zip(batched_preds, G_boxes, G_used_flags):
            pb, ps, pl = pred
            m = (pl == c)
            pb = pb[m]; ps = ps[m]
            if len(pb) == 0:
                continue
            # map local indices to global order positions
            local_order = np.argsort(-ps)
            pb = pb[local_order]

            gb = gt
            used_flags = used
            ious = box_iou_xyxy(pb, gb)  # [Np, Ng]
            for j in range(pb.shape[0]):
                if gb.shape[0] == 0:
                    fp[idx] = 1.0; idx += 1; continue
                i = np.argmax(ious[j])
                if ious[j, i] >= iou_th and not used_flags[i]:
                    tp[idx] = 1.0; used_flags[i] = True
                else:
                    fp[idx] = 1.0
                idx += 1

        # precision-recall & AP (11-pt interp is fine for a quick smoke test)
        if len(tp) == 0:
            aps[c]=0.0; prec[c]=0.0; rec[c]=0.0; continue
        cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
        recall = cum_tp / max(1, total_gts)
        precision = cum_tp / np.maximum(1, (cum_tp + cum_fp))
        rec[c] = float(recall[-1]); prec[c] = float(precision[np.argmax(recall)])  # crude
        # AP as trapezoidal area
        aps[c] = float(np.trapz(precision, recall))

    mAP = float(aps.mean())
    return mAP, aps, prec, rec