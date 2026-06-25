from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **_kw):
        return it

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def center_crop(img: Image.Image, size: int) -> Image.Image:
    """Centered `size x size` crop. If a side is shorter than `size`, the crop is
    padded with black so the output is always exactly size x size."""
    w, h = img.size
    if w >= size and h >= size:
        left = (w - size) // 2
        top = (h - size) // 2
        return img.crop((left, top, left + size, top + size))
    # rare: image smaller than crop on some side -> paste centered onto black canvas
    left = (w - size) // 2
    top = (h - size) // 2
    cropped = img.crop((left, top, left + size, top + size))  # PIL pads OOB with 0
    return cropped


def process_one(src: Path, in_root: Path, out_root: Path, size: int, overwrite: bool) -> str:
    rel = src.relative_to(in_root)
    dst = out_root / rel
    if dst.exists() and not overwrite:
        return "skip"
    try:
        with Image.open(src) as im:
            im.load()
            out = center_crop(im, size)
            dst.parent.mkdir(parents=True, exist_ok=True)
            # preserve format by extension; PNG is lossless, JPEG keeps source quality
            save_kwargs = {}
            if dst.suffix.lower() in (".jpg", ".jpeg"):
                save_kwargs = {"quality": 95}
            out.save(dst, **save_kwargs)
        return "ok"
    except Exception as e:  # noqa: BLE001 - report and continue
        return f"err:{src.name}:{e}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Center-crop a tree of images to SIZE x SIZE")
    p.add_argument("--input-dir", type=Path, required=True, help="root folder of resized images")
    p.add_argument("--output-dir", type=Path, required=True, help="mirror output folder (new)")
    p.add_argument("--size", type=int, default=448, help="square crop size (default 448)")
    p.add_argument("--workers", type=int, default=8, help="parallel worker threads")
    p.add_argument("--overwrite", action="store_true", help="re-crop even if output exists")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_root: Path = args.input_dir.resolve()
    out_root: Path = args.output_dir.resolve()
    if not in_root.is_dir():
        raise SystemExit(f"[ERROR] input-dir not found: {in_root}")
    if out_root == in_root:
        raise SystemExit("[ERROR] output-dir must differ from input-dir (writes a new tree)")

    print(f"Scanning {in_root} for images ...")
    files = [p for p in in_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    print(f"  found {len(files):,} images -> cropping {args.size}x{args.size} into {out_root}")
    if not files:
        return 0

    ok = skipped = errors = 0
    err_samples: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, f, in_root, out_root, args.size, args.overwrite) for f in files]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="cropping", unit="img"):
            r = fut.result()
            if r == "ok":
                ok += 1
            elif r == "skip":
                skipped += 1
            else:
                errors += 1
                if len(err_samples) < 10:
                    err_samples.append(r)

    print(f"\nDONE. cropped={ok:,}  skipped(existing)={skipped:,}  errors={errors:,}")
    for e in err_samples:
        print("  ", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
