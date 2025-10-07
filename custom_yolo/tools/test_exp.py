from pathlib import Path; import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolo.exp import Exp

if __name__ == "__main__":
    exp = Exp()
    exp.data_num_workers = 0  # avoid spawn quirks on macOS for this test

    loader = exp.get_data_loader(batch_size=2, is_distributed=False, no_aug=True, cache_img=None)
    batch = next(iter(loader))

    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            imgs, targets = batch
            img_info = img_ids = None
        elif len(batch) == 4:
            imgs, targets, img_info, img_ids = batch
        else:
            raise ValueError(f"Unexpected batch format with len={len(batch)}")
    else:
        raise ValueError(f"Unexpected batch type: {type(batch)}")

    print("imgs:", imgs.shape, imgs.dtype)
    print("targets:", targets.shape, targets.dtype)
    if img_info is not None:
        print("img_info[0]:", img_info[0])