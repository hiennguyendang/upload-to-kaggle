"""Build silver-style scene-graph JSON for the ImaGenome **gold** set (~1000 images).

The gold release ships structured *.txt/.csv* instead of the nice per-image JSON the
silver set has. This reassembles them into the SAME schema silver uses
(`objects` + `attributes` + `relationships`), one `<dicom>_SceneGraph.json` per image,
so the gold set drops straight into the phase_2 detector / LLM pipeline as the
held-out test split.

Sources (all under gold_dataset/):
  - gold_bbox_coordinate_annotations_1000images.csv   -> objects[] (bbox coords, incl original_*)
  - gold_object_attribute_with_coordinates.txt        -> attributes[] (per-bbox findings)
  - gold_object_comparison_with_coordinates.txt       -> relationships[] + comparison_cues
  - mimic metadata csv                                -> viewpoint / patient_id / study_id

Output is in the silver **raw** coordinate convention (224-space `x1..y2` + full-res
`original_*`). Run `rescale.py` on the output dir afterwards to map into 448-crop space,
exactly like the silver graphs.

python preprocess/scene_graph/build_gold_scene_graph.py --gold-dir "C:\\Users\\Dang Hien\\Downloads\\chest-imagenome-dataset-1.0.0\\gold_dataset" --metadata data\\mimic-cxr-2.0.0-metadata.csv --out-dir "C:\\Users\\Dang Hien\\Downloads\\chest-imagenome-dataset-1.0.0\\gold_dataset"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it

csv.field_size_limit(10_000_000)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = REPO_ROOT / "data" / "mimic-cxr-2.0.0-metadata.csv"
DEFAULT_GOLD = Path(r"C:\Users\Dang Hien\Downloads\chest-imagenome-dataset-1.0.0\gold_dataset")
DEFAULT_SILVER = Path(
    r"C:\Users\Dang Hien\Downloads\chest-imagenome-dataset-1.0.0\silver_dataset\scene_graph"
)

BBOX_CSV = "gold_bbox_coordinate_annotations_1000images.csv"
ATTR_TXT = "gold_object_attribute_with_coordinates.txt"
CMP_TXT = "gold_object_comparison_with_coordinates.txt"

# comparison string -> human predicate (matches silver's `predicate` field style)
PREDICATE_MAP = {
    "no change": "No status change",
    "improved": "Improved",
    "worsened": "Worsened",
}


def stem(image_id: str) -> str:
    """dicom_id (strip .dcm if present)."""
    return image_id[:-4] if image_id.lower().endswith(".dcm") else image_id


def to_int(v, default=None):
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


# ---- bbox_name -> {synsets,name} lookup, harvested from a few silver graphs -----
def build_name_lut(silver_root: Path, sample: int = 400) -> dict[str, dict]:
    lut: dict[str, dict] = {}
    if not silver_root or not silver_root.is_dir():
        return lut
    for i, fp in enumerate(sorted(silver_root.glob("*.json"))):
        if i >= sample:
            break
        try:
            d = json.loads(fp.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        for o in d.get("objects", []):
            bn = o.get("bbox_name")
            if bn and bn not in lut:
                lut[bn] = {"synsets": o.get("synsets", []), "name": o.get("name", bn)}
    return lut


# ---- metadata: dicom -> image-level fields -------------------------------------
def load_meta(metadata_path: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    with open(metadata_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            d = row.get("dicom_id")
            if d:
                meta[d] = {
                    "viewpoint": (row.get("ViewPosition") or "").strip() or None,
                    "patient_id": to_int(row.get("subject_id")),
                    "study_id": to_int(row.get("study_id")),
                }
    return meta


# ---- objects from the bbox csv -------------------------------------------------
def load_objects(bbox_csv: Path, lut: dict[str, dict]) -> "OrderedDict[str, list]":
    by_image: "OrderedDict[str, list]" = OrderedDict()
    with open(bbox_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            img = row.get("image_id", "")
            bn = (row.get("bbox_name") or "").strip()
            if not img or not bn:
                continue
            sid = stem(img)  # key everything by bare dicom_id (no .dcm)
            meta = lut.get(bn, {})
            x1, y1, x2, y2 = (to_int(row.get(k)) for k in ("x1", "y1", "x2", "y2"))
            obj = {
                "object_id": f"{sid}_{bn}",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": to_int(row.get("width")), "height": to_int(row.get("height")),
                "bbox_name": bn,
                "synsets": meta.get("synsets", []),
                "name": meta.get("name", bn),
                "original_x1": to_int(row.get("original_x1")),
                "original_y1": to_int(row.get("original_y1")),
                "original_x2": to_int(row.get("original_x2")),
                "original_y2": to_int(row.get("original_y2")),
                "original_width": to_int(row.get("original_width")),
                "original_height": to_int(row.get("original_height")),
            }
            by_image.setdefault(sid, []).append(obj)
    return by_image


# ---- attributes from the attribute txt -----------------------------------------
def load_attributes(attr_txt: Path, lut: dict[str, dict]) -> dict:
    """-> {image_id: {bbox_name: attr_entry}} in silver shape."""
    # gather raw rows grouped by (image, bbox), preserving phrase order
    phrases: dict = defaultdict(lambda: OrderedDict())  # (img,bbox) -> {(row_id,sent): [attr,...]}
    sections: dict = {}                                  # (img,bbox,row_id,sent) -> section
    with open(attr_txt, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            img, bbox = stem(row.get("image_id", "")), (row.get("bbox") or "").strip()
            attr = (row.get("attribute") or "").strip()
            if not img or not bbox or not attr:
                continue
            row_id = (row.get("row_id") or "").strip()
            sent = (row.get("sentence") or "").strip()
            key = (row_id, sent)
            phrases[(img, bbox)].setdefault(key, [])
            if attr not in phrases[(img, bbox)][key]:
                phrases[(img, bbox)][key].append(attr)
            sections[(img, bbox, row_id, sent)] = (row.get("section") or "finalreport").strip()

    out: dict = defaultdict(dict)
    for (img, bbox), pmap in phrases.items():
        sid = stem(img)
        meta = lut.get(bbox, {})
        entry = {
            bbox: True,
            "bbox_name": bbox,
            "synsets": meta.get("synsets", []),
            "name": meta.get("name", bbox),
            "attributes": [],
            "attributes_ids": [],
            "phrases": [],
            "phrase_IDs": [],
            "sections": [],
            "comparison_cues": [],
            "temporal_cues": [],
            "severity_cues": [],
            "texture_cues": [],
            "object_id": f"{sid}_{bbox}",
        }
        for (row_id, sent), attrs in pmap.items():
            entry["attributes"].append(attrs)
            entry["attributes_ids"].append(["" for _ in attrs])
            entry["phrases"].append(sent)
            entry["phrase_IDs"].append(row_id)
            entry["sections"].append(sections.get((img, bbox, row_id, sent), "finalreport"))
            entry["comparison_cues"].append([])
            entry["temporal_cues"].append([])
            entry["severity_cues"].append([])
            entry["texture_cues"].append([])
        out[img][bbox] = entry
    return out


# ---- comparisons -> relationships[] + comparison_cues injected into attributes --
def split_cmp(comparison: str) -> list[str]:
    return [c.strip() for c in (comparison or "").split(";;") if c.strip()]


def load_comparisons(cmp_txt: Path):
    """-> (relationships_by_image, cue_phrases_by_image_bbox).
    cue phrases get merged into the attribute entries so comparison_cues are populated."""
    rels: dict = defaultdict(lambda: OrderedDict())  # img -> {rel_id: rel_entry}
    cues: dict = defaultdict(lambda: defaultdict(OrderedDict))  # img -> bbox -> {sent: data}
    with open(cmp_txt, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            img = stem((row.get("current_image_id") or "").strip())
            bbox = (row.get("bbox") or "").strip()
            comp = (row.get("comparison") or "").strip()
            if not img or not bbox or not comp:
                continue
            rid = (row.get("relationship_id") or "").strip()
            attr = (row.get("attribute") or "").strip()
            sent = (row.get("sentence") or "").strip()
            cue_names = [f"comparison|yes|{c}" for c in split_cmp(comp)]

            # relationship entry (group rows sharing relationship_id)
            r = rels[img].get(rid)
            if r is None:
                r = {
                    "relationship_id": rid,
                    "predicate": "['" + "', '".join(
                        PREDICATE_MAP.get(c, c) for c in split_cmp(comp)) + "']",
                    "synsets": [],
                    "relationship_names": list(cue_names),
                    "relationship_contexts": [1.0 for _ in cue_names],
                    "phrase": sent,
                    "attributes": [],
                    "bbox_name": bbox,
                    "subject_id": (row.get("subject_id") or "").strip(),
                    "object_id": (row.get("object_id") or "").strip(),
                }
                rels[img][rid] = r
            if attr and attr not in r["attributes"]:
                r["attributes"].append(attr)

            # cue phrase to merge into attributes[img][bbox]
            slot = cues[img][bbox].setdefault(sent, {"attrs": [], "cues": [], "rid": rid})
            if attr and attr not in slot["attrs"]:
                slot["attrs"].append(attr)
            for cn in cue_names:
                if cn not in slot["cues"]:
                    slot["cues"].append(cn)
    return rels, cues


def merge_cues_into_attributes(attrs_by_image: dict, cues_by_image: dict, lut: dict):
    for img, bbmap in cues_by_image.items():
        for bbox, sents in bbmap.items():
            sid = stem(img)
            entry = attrs_by_image[img].get(bbox)
            if entry is None:
                meta = lut.get(bbox, {})
                entry = {
                    bbox: True, "bbox_name": bbox,
                    "synsets": meta.get("synsets", []), "name": meta.get("name", bbox),
                    "attributes": [], "attributes_ids": [], "phrases": [], "phrase_IDs": [],
                    "sections": [], "comparison_cues": [], "temporal_cues": [],
                    "severity_cues": [], "texture_cues": [], "object_id": f"{sid}_{bbox}",
                }
                attrs_by_image[img][bbox] = entry
            for sent, data in sents.items():
                # if this sentence already a phrase, merge cues there; else add new phrase
                if sent in entry["phrases"]:
                    idx = entry["phrases"].index(sent)
                    for cn in data["cues"]:
                        if cn not in entry["comparison_cues"][idx]:
                            entry["comparison_cues"][idx].append(cn)
                else:
                    entry["attributes"].append(list(data["attrs"]))
                    entry["attributes_ids"].append(["" for _ in data["attrs"]])
                    entry["phrases"].append(sent)
                    entry["phrase_IDs"].append(data["rid"])
                    entry["sections"].append("finalreport")
                    entry["comparison_cues"].append(list(data["cues"]))
                    entry["temporal_cues"].append([])
                    entry["severity_cues"].append([])
                    entry["texture_cues"].append([])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild silver-style JSON for the ImaGenome gold set")
    p.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD, help="gold_dataset folder")
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="mimic metadata csv")
    p.add_argument("--silver-root", type=Path, default=DEFAULT_SILVER,
                   help="silver scene_graph dir (only to harvest bbox synsets/names)")
    p.add_argument("--out-dir", type=Path, required=True, help="output dir (raw 224-space JSON)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bbox_csv = args.gold_dir / BBOX_CSV
    attr_txt = args.gold_dir / ATTR_TXT
    cmp_txt = args.gold_dir / CMP_TXT
    for pth in (bbox_csv, attr_txt, cmp_txt, args.metadata):
        if not pth.exists():
            raise SystemExit(f"[ERROR] missing input: {pth}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("1. bbox_name -> synset/name lookup from silver ...")
    lut = build_name_lut(args.silver_root)
    print(f"   -> {len(lut)} region names mapped"
          + ("" if lut else "  (silver not found; using bbox_name only)"))

    print("2. metadata (viewpoint / ids) ...")
    meta = load_meta(args.metadata)
    print(f"   -> {len(meta):,} dicom rows")

    print("3. objects from bbox csv ...")
    objects_by_image = load_objects(bbox_csv, lut)
    print(f"   -> {len(objects_by_image):,} images with bboxes")

    print("4. attributes ...")
    attrs_by_image = load_attributes(attr_txt, lut)

    print("5. comparisons -> relationships + cues ...")
    rels_by_image, cues_by_image = load_comparisons(cmp_txt)
    merge_cues_into_attributes(attrs_by_image, cues_by_image, lut)

    print("6. writing scene graphs ...")
    written = 0
    n_attr = n_rel = 0
    for img, objects in tqdm(objects_by_image.items(), total=len(objects_by_image), unit="img"):
        sid = stem(img)
        m = meta.get(sid, {})
        attrs = list(attrs_by_image.get(img, {}).values())
        rels = list(rels_by_image.get(img, {}).values())
        n_attr += sum(1 for a in attrs if any(a["attributes"]))
        n_rel += len(rels)
        scene = {
            "image_id": sid,
            "viewpoint": m.get("viewpoint"),
            "patient_id": m.get("patient_id"),
            "study_id": m.get("study_id"),
            "gender": None,
            "age_decile": None,
            "reason_for_exam": None,
            "StudyOrder": None,
            "StudyDateTime": None,
            "objects": objects,
            "attributes": attrs,
            "relationships": rels,
        }
        (args.out_dir / f"{sid}_SceneGraph.json").write_text(
            json.dumps(scene, ensure_ascii=False, indent=4), encoding="utf-8"
        )
        written += 1

    print(f"\n[DONE] {written:,} gold scene graphs -> {args.out_dir}")
    print(f"       bbox entries written; ~{n_attr:,} bbox-attribute groups, {n_rel:,} relationships")
    print("       next: run rescale.py on this out-dir to map bboxes into 448-crop space.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
