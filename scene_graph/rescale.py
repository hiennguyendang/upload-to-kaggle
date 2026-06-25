"""Rescale ImaGenome bboxes into the **448x448 center-crop** coordinate space.

The image pipeline is: resize short-side -> 512 (aspect preserved, BILINEAR) then
center-crop 448x448. This maps every object's full-resolution `original_x1..y2`
through that exact same geometry, overwriting `x1,y1,x2,y2` (+ `width,height`).

Geometry is kept identical to `preprocess/center_crop_images.py`:
  scale          = 512 / min(src_w, src_h)              # uniform, matches PIL resize
  new_w, new_h   = round(src * scale)                   # resized canvas (short side = 512)
  crop_x0, crop_y0 = (new_w-448)//2, (new_h-448)//2     # integer floor == PIL .crop() origin
A box pushed fully outside the 448 window collapses to the (0,0,0,0) sentinel,
which downstream code already filters.

Reads image dimensions from the MIMIC metadata CSV (`Columns`=width, `Rows`=height),
keyed by `dicom_id`. Works for BOTH silver and gold scene graphs (same JSON shape).
Writes to a NEW output dir (non-destructive); resumable via a checkpoint file.

python preprocess/scene_graph/rescale.py --scene-dir  "C:\\Users\\Dang Hien\\Downloads\\chest-imagenome-dataset-1.0.0\\silver_dataset\\scene_graph" --out-dir "C:\\Users\\Dang Hien\\Downloads\\chest-imagenome-dataset-1.0.0\\silver_dataset\\scene_graph" --metadata data\\mimic-cxr-2.0.0-metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # tqdm optional
    def tqdm(it, **_kw):
        return it

# ---- repo defaults (override with flags) -------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = REPO_ROOT / "data" / "mimic-cxr-2.0.0-metadata.csv"
DEFAULT_SCENE_DIR = Path(
    r"C:\Users\Dang Hien\Downloads\chest-imagenome-dataset-1.0.0\silver_dataset\scene_graph"
)

TARGET_SHORT_SIDE = 512.0
CROP_SIZE = 448


def resized_canvas(src_w: int, src_h: int, short: float = TARGET_SHORT_SIDE) -> tuple[float, int, int]:
    """Return (uniform_scale, new_w, new_h) matching `short_edge_resize` exactly."""
    scale = short / min(src_w, src_h)
    if src_w < src_h:
        new_w = int(short)
        new_h = max(1, int(round(src_h * scale)))
    else:
        new_h = int(short)
        new_w = max(1, int(round(src_w * scale)))
    return scale, new_w, new_h


def transform_bbox(ox1, oy1, ox2, oy2, src_w, src_h, size: int = CROP_SIZE):
    """Map a full-resolution bbox into the centered size x size crop. Returns ints;
    (0,0,0,0) means the box fell entirely outside the crop window."""
    scale, new_w, new_h = resized_canvas(src_w, src_h)
    crop_x0 = (new_w - size) // 2
    crop_y0 = (new_h - size) // 2

    x1 = ox1 * scale - crop_x0
    y1 = oy1 * scale - crop_y0
    x2 = ox2 * scale - crop_x0
    y2 = oy2 * scale - crop_y0

    # clip to crop window then round to integer pixels
    fx1 = int(round(max(0.0, min(float(size), x1))))
    fy1 = int(round(max(0.0, min(float(size), y1))))
    fx2 = int(round(max(0.0, min(float(size), x2))))
    fy2 = int(round(max(0.0, min(float(size), y2))))

    if fx1 >= fx2 or fy1 >= fy2:  # box vanished after crop
        return 0, 0, 0, 0
    return fx1, fy1, fx2, fy2


def load_image_sizes(metadata_path: Path) -> dict[str, tuple[int, int]]:
    sizes: dict[str, tuple[int, int]] = {}
    with open(metadata_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dicom = row.get("dicom_id")
            cols, rows = row.get("Columns"), row.get("Rows")
            if dicom and cols and rows:
                try:
                    sizes[dicom] = (int(cols), int(rows))
                except ValueError:
                    continue
    return sizes


def dicom_id_of(scene: dict, json_path: Path) -> str:
    """The scene-graph filename stem == dicom_id (== metadata dicom_id). `image_id`
    inside the JSON is also the dicom_id for silver/gold ImaGenome."""
    iid = str(scene.get("image_id", "")).strip()
    if iid:
        return iid
    return json_path.name.replace("_SceneGraph.json", "").replace(".json", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rescale ImaGenome bboxes to 448 center-crop space")
    p.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR, help="input *_SceneGraph.json dir")
    p.add_argument("--out-dir", type=Path, required=True, help="output dir (new; non-destructive)")
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="mimic metadata csv")
    p.add_argument("--size", type=int, default=CROP_SIZE, help="center-crop size (default 448)")
    p.add_argument("--reset", action="store_true", help="ignore checkpoint, reprocess all")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.scene_dir.is_dir():
        raise SystemExit(f"[ERROR] scene-dir not found: {args.scene_dir}")
    if not args.metadata.exists():
        raise SystemExit(f"[ERROR] metadata csv not found: {args.metadata}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "_rescale_checkpoint.txt"

    print(f"1. Loading image sizes from {args.metadata} ...")
    sizes = load_image_sizes(args.metadata)
    print(f"   -> {len(sizes):,} image dimensions loaded.")

    done: set[str] = set()
    if checkpoint.exists() and not args.reset:
        done = {ln.strip() for ln in checkpoint.read_text(encoding="utf-8").splitlines() if ln.strip()}
        print(f"2. Checkpoint: {len(done):,} files already done -> skipping them.")
    elif args.reset and checkpoint.exists():
        checkpoint.unlink()

    all_files = sorted(args.scene_dir.glob("*.json"))
    todo = [f for f in all_files if f.name not in done]
    print(f"3. Rescaling {len(todo):,} / {len(all_files):,} scene graphs -> {args.out_dir}\n")

    missing = collapsed = written = 0
    with open(checkpoint, "a", encoding="utf-8") as ck:
        for jf in tqdm(todo, desc="rescale", unit="file"):
            try:
                data = json.loads(jf.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError) as e:
                tqdm.write(f"[SKIP] {jf.name}: {e}")
                continue
            dicom = dicom_id_of(data, jf)
            wh = sizes.get(dicom)
            if wh is None:
                missing += 1
                tqdm.write(f"[WARN] no metadata size for {dicom} ({jf.name}); skipped")
                continue
            src_w, src_h = wh
            for obj in data.get("objects", []):
                ox1, oy1 = obj.get("original_x1"), obj.get("original_y1")
                ox2, oy2 = obj.get("original_x2"), obj.get("original_y2")
                if None in (ox1, oy1, ox2, oy2):
                    continue
                nx1, ny1, nx2, ny2 = transform_bbox(ox1, oy1, ox2, oy2, src_w, src_h, args.size)
                obj["x1"], obj["y1"], obj["x2"], obj["y2"] = nx1, ny1, nx2, ny2
                obj["width"], obj["height"] = nx2 - nx1, ny2 - ny1  # keep consistent
                if (nx1, ny1, nx2, ny2) == (0, 0, 0, 0):
                    collapsed += 1
            (args.out_dir / jf.name).write_text(
                json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8"
            )
            ck.write(jf.name + "\n")
            ck.flush()
            written += 1

    print(f"\n[DONE] written={written:,}  missing_size={missing:,}  boxes_collapsed={collapsed:,}")
    print(f"       output -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
