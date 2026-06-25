"""Fix the CheXplus split so the IMAGE-level ratio is ~70/10/20.

Problem: build_chexplus_splits.py picks val/test from the "gold" tier by PATIENT
count, but gold patients are image-heavy (~6 img each vs ~1.6 for silver), so 20%
of patients became ~42% of images. This re-picks val/test from gold patients by an
IMAGE budget (keeps the "eval on temporally-rich cases" intent, fixes the ratio),
then patches the `split` field in chexplpus_metadata_final.jsonl. Non-destructive:
backs up both files to .bak first.

    python preprocess/scene_graph/resplit_chexplus_by_images.py \
        --splits data/chexplus_splits.csv --metadata data/chexplpus_metadata_final.jsonl \
        --test-frac 0.20 --val-frac 0.10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

PAT_RE = re.compile(r"patient\d+")


def patient_of(row: dict) -> str:
    p = str(row.get("patient_id", "")).strip()
    if p:
        return p
    m = PAT_RE.search(str(row.get("image_id", "")))
    return m.group(0) if m else ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-split CheXplus by image budget")
    p.add_argument("--splits", type=Path, default=Path("data/chexplus_splits.csv"))
    p.add_argument("--metadata", type=Path, default=Path("data/chexplpus_metadata_final.jsonl"))
    p.add_argument("--test-frac", type=float, default=0.20)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="report ratios, write nothing")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for f in (args.splits, args.metadata):
        if not f.exists():
            raise SystemExit(f"[ERROR] not found: {f}")

    # patient -> tier (+ keep the other columns to rewrite the csv)
    tier, extra_cols, rows_csv = {}, [], []
    with open(args.splits, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        extra_cols = reader.fieldnames or []
        for r in reader:
            tier[r["patient_id"]] = r.get("patient_tier", "silver")
            rows_csv.append(r)

    # images per patient + current split counts
    imgs_per_pat: Counter = Counter()
    cur_split_imgs: Counter = Counter()
    total = 0
    for line in open(args.metadata, encoding="utf-8-sig"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        pat = patient_of(row)
        imgs_per_pat[pat] += 1
        cur_split_imgs[str(row.get("split", "?"))] += 1
        total += 1

    gold = sorted(p for p in tier if tier.get(p) == "gold")
    random.Random(args.seed).shuffle(gold)
    test_budget = total * args.test_frac
    val_budget = total * args.val_frac

    new_split: dict[str, str] = {p: "train" for p in tier}  # default train
    acc = 0
    it = iter(gold)
    for p in it:                       # fill TEST first
        new_split[p] = "test"; acc += imgs_per_pat.get(p, 0)
        if acc >= test_budget:
            break
    acc = 0
    for p in it:                       # then VAL from remaining gold
        new_split[p] = "val"; acc += imgs_per_pat.get(p, 0)
        if acc >= val_budget:
            break
    # everything else (gold leftovers + all silver) stays "train"

    # report
    new_img = Counter()
    new_pat = Counter()
    for p, sp in new_split.items():
        new_img[sp] += imgs_per_pat.get(p, 0)
        new_pat[sp] += 1
    def show(name, c, tot):
        return " | ".join(f"{k} {c.get(k,0):,} ({100*c.get(k,0)/max(1,tot):.1f}%)"
                          for k in ("train", "val", "test"))
    print(f"images total: {total:,}")
    print(f"BEFORE images: {show('img', cur_split_imgs, total)}")
    print(f"AFTER  images: {show('img', new_img, total)}")
    print(f"AFTER  patients: {show('pat', new_pat, sum(new_pat.values()))}")
    if args.dry_run:
        print("[dry-run] nothing written"); return 0

    # rewrite splits.csv (backup first)
    shutil.copyfile(args.splits, args.splits.with_suffix(".csv.bak"))
    with open(args.splits, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=extra_cols)
        w.writeheader()
        for r in rows_csv:
            r["split"] = new_split.get(r["patient_id"], "train")
            w.writerow(r)

    # patch metadata split field (stream to temp, then replace; backup first)
    shutil.copyfile(args.metadata, args.metadata.with_suffix(".jsonl.bak"))
    tmp = args.metadata.with_suffix(".jsonl.tmp")
    changed = 0
    with open(args.metadata, encoding="utf-8-sig") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sp = new_split.get(patient_of(row), "train")
            if row.get("split") != sp:
                changed += 1
            row["split"] = sp
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, args.metadata)
    print(f"\n[DONE] patched {changed:,} image splits. Backups: *.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
