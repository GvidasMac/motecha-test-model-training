import contextlib, io, itertools, json, tempfile, time
from collections import ChainMap, defaultdict
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm
import numpy as np
import torch

from pycocotools.cocoeval import COCOeval  # always use pycocotools

from yolo.utils import (
    gather, is_main_process, postprocess, synchronize,
    time_synchronized, xyxy2xywh
)

def _per_class_table(coco_eval, class_names, metric="AP", columns=6):
    if metric == "AP":
        precisions = coco_eval.eval["precision"]  # TxRxKxAxM
        vals = {}
        for k, name in enumerate(class_names):
            p = precisions[:, :, k, 0, -1]
            p = p[p > -1]
            vals[name] = float(np.mean(p) * 100) if p.size else float("nan")
    else:
        recalls = coco_eval.eval["recall"]  # TxKxAxM
        vals = {}
        for k, name in enumerate(class_names):
            r = recalls[:, k, 0, -1]
            r = r[r > -1]
            vals[name] = float(np.mean(r) * 100) if r.size else float("nan")

    headers = ["class", metric]
    num_cols = min(columns, len(vals) * len(headers))
    result_pair = [x for pair in vals.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    return tabulate(row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left")

class COCOEvaluator:
    def __init__(self, dataloader, img_size, confthre, nmsthre, num_classes, testdev=False,
                 per_class_AP=True, per_class_AR=True):
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR

    def evaluate(self, model, is_distributed=False, half=False, return_outputs=False, **_):
        model.eval()
        device = next(model.parameters()).device
        dtype = torch.float16 if (half and device.type == "cuda") else torch.float32

        data_list = []
        output_data = defaultdict()
        pbar = tqdm(self.dataloader) if is_main_process() else self.dataloader

        inference_time = 0.0
        nms_time = 0.0
        n_samples = max(len(self.dataloader) - 1, 1)

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(pbar):
            with torch.no_grad():
                imgs = imgs.to(device=device, dtype=dtype, non_blocking=False)

                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            dl, image_wise = self.convert_to_coco_format(outputs, info_imgs, ids, return_outputs=True)
            data_list.extend(dl)
            output_data.update(image_wise)

        statistics = torch.tensor([inference_time, nms_time, n_samples], device=device, dtype=torch.float32)

        if is_distributed:
            synchronize()
            data_list = gather(data_list, dst=0)
            output_data = gather(output_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            output_data = dict(ChainMap(*output_data))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return (eval_results, output_data) if return_outputs else eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids, return_outputs=False):
        data_list = []
        image_wise_data = defaultdict(dict)

        for (output, img_h, img_w, img_id) in zip(outputs, info_imgs[0], info_imgs[1], ids):
            if output is None:
                continue
            output = output.cpu()
            bboxes = output[:, 0:4]
            # undo resize
            scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            # categories back to original COCO ids
            cat_ids = [self.dataloader.dataset.class_ids[int(c.item())] for c in cls]

            image_wise_data[int(img_id)] = {
                "bboxes": bboxes.numpy().tolist(),
                "scores": scores.numpy().tolist(),
                "categories": cat_ids,
            }

            bboxes_xywh = xyxy2xywh(bboxes)
            for ind in range(bboxes_xywh.shape[0]):
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": cat_ids[ind],
                    "bbox": bboxes_xywh[ind].numpy().tolist(),
                    "score": float(scores[ind].item()),
                    "segmentation": [],
                }
                data_list.append(pred_data)

        return (data_list, image_wise_data) if return_outputs else data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0.0, 0.0, None

        logger.info("Evaluate in main process...")

        inference_time, nms_time, n_samples = [x.item() for x in statistics]
        a_infer = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms = 1000 * nms_time / (n_samples * self.dataloader.batch_size)
        info = f"Average forward time: {a_infer:.2f} ms, Average NMS time: {a_nms:.2f} ms, " \
               f"Average inference time: {(a_infer + a_nms):.2f} ms\n"

        if len(data_dict) == 0:
            return 0.0, 0.0, info

        cocoGt = self.dataloader.dataset.coco
        _, tmp = tempfile.mkstemp()
        json.dump(data_dict, open(tmp, "w"))
        cocoDt = cocoGt.loadRes(tmp)

        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        info += redirect_string.getvalue()

        # per-class tables
        cat_ids = list(cocoGt.cats.keys())
        class_names = [cocoGt.cats[catId]["name"] for catId in sorted(cat_ids)]
        try:
            info += "per class AP:\n" + _per_class_table(cocoEval, class_names, "AP") + "\n"
            info += "per class AR:\n" + _per_class_table(cocoEval, class_names, "AR") + "\n"
        except Exception:
            pass

        return float(cocoEval.stats[0]), float(cocoEval.stats[1]), info