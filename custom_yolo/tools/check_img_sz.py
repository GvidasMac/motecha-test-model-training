from pathlib import Path
from collections import Counter, defaultdict
import argparse
import sys

# Pillow for image size reading
try:
    from PIL import Image
except ImportError:
    print("This script requires Pillow. Install it with:\n  pip install pillow", file=sys.stderr)
    sys.exit(1)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def iter_images(dir_path: Path):
    """Yield image file paths under dir_path with known extensions."""
    if not dir_path.exists():
        return
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def get_image_size(path: Path):
    """Return (width, height) or None if unreadable."""
    try:
        with Image.open(path) as im:
            return im.size  # (width, height)
    except Exception:
        return None

def scan_dir(dir_path: Path):
    """Scan one directory, return (Counter of (w,h), errors list)."""
    sizes = Counter()
    errors = []
    for img in iter_images(dir_path):
        sz = get_image_size(img)
        if sz is None:
            errors.append(str(img))
        else:
            sizes[sz] += 1
    return sizes, errors

def main():
    parser = argparse.ArgumentParser(description="List unique image resolutions in a dataset.")
    parser.add_argument("--data-root", default="/Users/gvmaka/projects/energesman-waste-detection-and-classification/dataset", type=Path)
    parser.add_argument("--train-images", default="images/train", type=Path)
    parser.add_argument("--val-images", default="images/val", type=Path)
    args = parser.parse_args()

    train_dir = args.data_root / args.train_images
    val_dir = args.data_root / args.val_images

    # Per-split scanning
    split_to_sizes = {}
    split_to_errors = {}
    for split_name, split_dir in (("train", train_dir), ("val", val_dir)):
        sizes, errors = scan_dir(split_dir)
        split_to_sizes[split_name] = sizes
        split_to_errors[split_name] = errors

    # Merge sizes across splits
    total_sizes = Counter()
    for sizes in split_to_sizes.values():
        total_sizes.update(sizes)

    # Print results
    def fmt_size(sz):
        w, h = sz
        return f"{w}x{h}"

    total_images = sum(total_sizes.values())
    print(f"\n=== Unique Image Resolutions (All Splits) ===")
    print(f"Total images scanned: {total_images}")
    print(f"Unique resolutions : {len(total_sizes)}\n")
    for (w, h), cnt in sorted(total_sizes.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{fmt_size((w, h)):<12}  count: {cnt}")

    # Optional: per-split breakdown
    for split_name, sizes in split_to_sizes.items():
        split_total = sum(sizes.values())
        print(f"\n--- {split_name.upper()} split ---")
        print(f"Images: {split_total}")
        if len(sizes) == 0:
            print("(no images found)")
            continue
        for (w, h), cnt in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"{fmt_size((w, h)):<12}  count: {cnt}")

    # Report unreadable files, if any
    all_errors = sum((split_to_errors[s] for s in split_to_errors), [])
    if all_errors:
        print("\nSome files could not be read (possibly corrupted or unsupported):")
        for p in all_errors:
            print(f"  {p}")

if __name__ == "__main__":
    main()