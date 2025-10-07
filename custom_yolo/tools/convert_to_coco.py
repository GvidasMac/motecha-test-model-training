import json, os, glob
from PIL import Image

# Edit these:
DATA_ROOT = "/Users/gvmaka/projects/energesman-waste-detection-and-classification/dataset"
SPLIT = "val"  # "train" or "val"
NUM_CLASSES = 6
CATEGORY_NAMES = [f"class_{i}" for i in range(NUM_CLASSES)]
OUT_JSON = os.path.join(DATA_ROOT, f"instances_{SPLIT}.json")

def load_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls, cx, cy, w, h = map(float, line.split())
            boxes.append((int(cls), cx, cy, w, h))
    return boxes

def yolo_to_xywh(cx, cy, w, h, W, H):
    # normalized -> pixel xywh
    x = (cx - w/2) * W
    y = (cy - h/2) * H
    return [x, y, w * W, h * H]

def main():
    img_dir = os.path.join(DATA_ROOT, "images", SPLIT)
    lbl_dir = os.path.join(DATA_ROOT, "labels", SPLIT)

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                       glob.glob(os.path.join(img_dir, "*.png")) +
                       glob.glob(os.path.join(img_dir, "*.jpeg")))

    images, annotations = [], []
    ann_id = 1
    for img_id, ip in enumerate(img_paths, start=1):
        file_name = os.path.relpath(ip, DATA_ROOT).replace("\\", "/")
        W, H = Image.open(ip).size
        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": W,
            "height": H
        })

        stem = os.path.splitext(os.path.basename(ip))[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        for (cls, cx, cy, w, h) in load_labels(lbl_path):
            x, y, bw, bh = yolo_to_xywh(cx, cy, w, h, W, H)
            if bw <= 0 or bh <= 0:  # skip degenerate boxes
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls,
                "bbox": [x, y, bw, bh],     # COCO xywh in pixels
                "area": bw * bh,
                "iscrowd": 0,
                "segmentation": []
            })
            ann_id += 1

    categories = [{"id": i, "name": CATEGORY_NAMES[i]} for i in range(NUM_CLASSES)]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(coco, f)
    print(f"Wrote {OUT_JSON} with {len(images)} images and {len(annotations)} annotations.")

if __name__ == "__main__":
    main()