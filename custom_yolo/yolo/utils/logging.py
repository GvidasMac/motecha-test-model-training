import logging, sys
from pathlib import Path
from tqdm import tqdm

def get_logger(log_path: Path):
    logger = logging.getLogger("custom_yolo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(fmt); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)
    return logger

class TqdmBar(tqdm):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)