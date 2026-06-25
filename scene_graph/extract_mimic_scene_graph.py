"""Flatten MIMIC ImaGenome scene-graph JSON files into a 2D CSV.

Output schema:
['patient_id', 'study_id', 'study_key', 'source_section', 'source_view',
 'anatomy', 'observation', 'presence', 'temporal_status', 'bboxes']

Notes:
- Only JSON files that physically exist in the scene_graph directory are processed.
- Bounding boxes are converted to 512x512 letterboxed coordinates using
  Rows/Columns from mimic-cxr-2.0.0-metadata.csv when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA = [
    "patient_id",
    "study_id",
    "study_key",
    "source_section",
    "source_view",
    "anatomy",
    "observation",
    "presence",
    "temporal_status",
    "bboxes",
]

DEFAULT_SCENE_GRAPH_DIR = Path(r"C:\Users\dhint\CHEX-DATA\MIMIC-CXR\ImaGenome\silver_dataset\scene_graph")
DEFAULT_METADATA_CSV = Path(r"C:\Users\dhint\CHEX-DATA\MIMIC-CXR\mimic-cxr-2.0.0-metadata.csv")
DEFAULT_OUTPUT_CSV = Path(r"C:\Users\dhint\CHEX-DATA\MIMIC-CXR\metadata\mimic_scene_graph.csv")
DEFAULT_OUTPUT_SIZE = 512

_TEMPORAL_ALIASES = {
    "stable": ["no change", "unchanged", "stable", "similar", "persistent", "persist"],
    "worsened": ["worsened", "worsening", "increased", "increase", "progression", "progressive"],
    "improved": ["improved", "improving", "decreased", "decrease", "resolved", "resolving"],
    "new": ["new", "newly", "developed", "developing", "emerging"],
}

_EXCLUDED_ANATOMY_KEYS = {
    "bbox_name",
    "synsets",
    "name",
    "attributes",
    "attributes_ids",
    "phrases",
    "phrase_ids",
    "sections",
    "comparison_cues",
    "temporal_cues",
    "severity_cues",
    "texture_cues",
    "object_id",
}

_DICOM_DIMS: dict[str, tuple[int, int]] | None = None
_DICOM_DIMS_PATH: Path | None = None


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("mimic_scene_graph")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_dicom_dimensions(metadata_csv: Path) -> dict[str, tuple[int, int]]:
    dims: dict[str, tuple[int, int]] = {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            dicom_id = _normalize_text(row.get("dicom_id"))
            rows = _safe_int(row.get("Rows"))
            cols = _safe_int(row.get("Columns"))
            if not dicom_id or not rows or not cols:
                continue
            dims[dicom_id] = (cols, rows)
    return dims


def _get_cached_dimensions(metadata_csv: Path = DEFAULT_METADATA_CSV) -> dict[str, tuple[int, int]]:
    global _DICOM_DIMS
    global _DICOM_DIMS_PATH

    metadata_csv = metadata_csv.resolve()
    if _DICOM_DIMS is None or _DICOM_DIMS_PATH != metadata_csv:
        _DICOM_DIMS = load_dicom_dimensions(metadata_csv)
        _DICOM_DIMS_PATH = metadata_csv
    return _DICOM_DIMS


def _letterbox_transform_bbox(
    bbox: tuple[float, float, float, float],
    src_width: int,
    src_height: int,
    output_size: int = DEFAULT_OUTPUT_SIZE,
) -> list[int]:
    x1, y1, x2, y2 = bbox

    if src_width <= 0 or src_height <= 0:
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

    scale = min(output_size / max(src_width, 1), output_size / max(src_height, 1))
    new_w = max(1, int(round(src_width * scale)))
    new_h = max(1, int(round(src_height * scale)))
    pad_x = (output_size - new_w) // 2
    pad_y = (output_size - new_h) // 2

    tx1 = int(round(x1 * scale)) + pad_x
    ty1 = int(round(y1 * scale)) + pad_y
    tx2 = int(round(x2 * scale)) + pad_x
    ty2 = int(round(y2 * scale)) + pad_y

    max_coord = output_size - 1
    tx1 = min(max(tx1, 0), max_coord)
    ty1 = min(max(ty1, 0), max_coord)
    tx2 = min(max(tx2, 0), max_coord)
    ty2 = min(max(ty2, 0), max_coord)

    return [tx1, ty1, tx2, ty2]


def _resolve_bbox_for_object(
    obj: dict[str, Any],
    src_dims: tuple[int, int] | None,
    output_size: int = DEFAULT_OUTPUT_SIZE,
) -> list[int] | None:
    src_width: int | None = None
    src_height: int | None = None
    if src_dims is not None:
        src_width, src_height = src_dims

    original_keys = ("original_x1", "original_y1", "original_x2", "original_y2")
    base_keys = ("x1", "y1", "x2", "y2")

    has_original = all(_safe_int(obj.get(k)) is not None for k in original_keys)
    has_base = all(_safe_int(obj.get(k)) is not None for k in base_keys)

    if has_original and src_width is not None and src_height is not None:
        bbox = tuple(float(_safe_int(obj.get(k))) for k in original_keys)
        return _letterbox_transform_bbox(bbox, src_width, src_height, output_size)

    if not has_base:
        return None

    x1 = float(_safe_int(obj.get("x1")))
    y1 = float(_safe_int(obj.get("y1")))
    x2 = float(_safe_int(obj.get("x2")))
    y2 = float(_safe_int(obj.get("y2")))

    # If base coordinates are already in low-resolution space (e.g. <= 512),
    # keep them as-is. Otherwise, treat them as original coordinates and
    # transform using source image dimensions when available.
    if max(x2, y2) <= output_size:
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

    if src_width is not None and src_height is not None:
        return _letterbox_transform_bbox((x1, y1, x2, y2), src_width, src_height, output_size)

    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def _extract_study_key(payload: dict[str, Any], json_path: Path) -> str:
    study_key = _normalize_text(payload.get("image_id"))
    if study_key:
        return study_key

    stem = json_path.stem
    if stem.endswith("_SceneGraph"):
        return stem[: -len("_SceneGraph")]
    return stem


def _extract_anatomy(attribute_entry: dict[str, Any]) -> str | None:
    bbox_name = _normalize_text(attribute_entry.get("bbox_name"))
    if bbox_name:
        return bbox_name

    for key, value in attribute_entry.items():
        if key.lower() in _EXCLUDED_ANATOMY_KEYS:
            continue
        if isinstance(value, bool) and value:
            return _normalize_text(key)

    name = _normalize_text(attribute_entry.get("name"))
    return name or None


def _normalize_presence(token: str) -> str | None:
    lowered = token.strip().lower()
    if lowered == "yes":
        return "present"
    if lowered == "no":
        return "absent"
    if lowered:
        return lowered
    return None


def _normalize_temporal_status(raw_status: str) -> str | None:
    text = raw_status.strip().lower()
    if not text:
        return None

    for mapped, aliases in _TEMPORAL_ALIASES.items():
        if any(alias in text for alias in aliases):
            return mapped
    return text


def _extract_temporal_status_from_cues(cues: Any) -> str | None:
    if not isinstance(cues, list):
        return None

    for cue in cues:
        cue_text = _normalize_text(cue)
        if not cue_text:
            continue
        parts = [part.strip() for part in cue_text.split("|")]
        if len(parts) >= 3:
            return _normalize_temporal_status(parts[2])
    return None


def _build_bbox_map(
    payload: dict[str, Any],
    src_dims: tuple[int, int] | None,
    output_size: int,
) -> dict[str, list[int] | None]:
    bbox_map: dict[str, list[int] | None] = {}
    objects = payload.get("objects")
    if not isinstance(objects, list):
        return bbox_map

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        bbox_name = _normalize_text(obj.get("bbox_name") or obj.get("name")).lower()
        if not bbox_name:
            continue
        bbox_map[bbox_name] = _resolve_bbox_for_object(obj, src_dims, output_size)
    return bbox_map


def process_mimic_json_rows(json_path: str) -> list[dict[str, Any]]:
    """Flatten one MIMIC scene-graph JSON into row dictionaries."""
    file_path = Path(json_path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a top-level JSON object in {file_path}")

    patient_id = payload.get("patient_id")
    study_id = payload.get("study_id")
    study_key = _extract_study_key(payload, file_path)
    source_view = _normalize_text(payload.get("viewpoint")) or None
    source_section = "finalreport"

    dicom_dims = _get_cached_dimensions().get(study_key)
    bbox_map = _build_bbox_map(payload, dicom_dims, DEFAULT_OUTPUT_SIZE)

    attributes = payload.get("attributes")
    if not isinstance(attributes, list):
        return []

    rows: list[dict[str, Any]] = []
    for entry in attributes:
        if not isinstance(entry, dict):
            continue

        anatomy = _extract_anatomy(entry)
        anatomy_key = _normalize_text(anatomy).lower() if anatomy else ""
        bbox_value = bbox_map.get(anatomy_key)

        grouped_attributes = entry.get("attributes")
        comparison_groups = entry.get("comparison_cues")

        if not isinstance(grouped_attributes, list):
            continue
        if not isinstance(comparison_groups, list):
            comparison_groups = []

        for idx, attribute_group in enumerate(grouped_attributes):
            if not isinstance(attribute_group, list):
                continue

            temporal_status = None
            if idx < len(comparison_groups):
                temporal_status = _extract_temporal_status_from_cues(comparison_groups[idx])

            for token in attribute_group:
                token_text = _normalize_text(token)
                if not token_text:
                    continue

                parts = [part.strip() for part in token_text.split("|")]
                if len(parts) < 3:
                    continue

                presence = _normalize_presence(parts[1])
                observation = "|".join(parts[2:]).strip()
                if not observation:
                    continue

                rows.append(
                    {
                        "patient_id": patient_id,
                        "study_id": study_id,
                        "study_key": study_key,
                        "source_section": source_section,
                        "source_view": source_view,
                        "anatomy": anatomy,
                        "observation": observation,
                        "presence": presence,
                        "temporal_status": temporal_status,
                        "bboxes": json.dumps(bbox_value) if bbox_value is not None else None,
                    }
                )

    return rows


def process_mimic_json(json_path: str) -> pd.DataFrame:
    """Flatten one MIMIC ImaGenome scene-graph JSON into a DataFrame.

    Parameters
    ----------
    json_path:
        Path to one scene-graph JSON file.
    """

    rows = process_mimic_json_rows(json_path)

    if not rows:
        return pd.DataFrame(columns=SCHEMA)

    df = pd.DataFrame(rows)
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = None
    return df[SCHEMA]


def process_scene_graph_directory(
    scene_graph_dir: Path,
    output_csv: Path,
    *,
    metadata_csv: Path = DEFAULT_METADATA_CSV,
    output_size: int = DEFAULT_OUTPUT_SIZE,
    max_files: int | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    if logger is None:
        logger = setup_logger()

    if not scene_graph_dir.exists() or not scene_graph_dir.is_dir():
        raise FileNotFoundError(f"Scene graph directory not found: {scene_graph_dir}")

    # Warm up global cache and size config used by process_mimic_json.
    global _DICOM_DIMS
    global _DICOM_DIMS_PATH
    _DICOM_DIMS = load_dicom_dimensions(metadata_csv)
    _DICOM_DIMS_PATH = metadata_csv.resolve()

    global DEFAULT_OUTPUT_SIZE
    DEFAULT_OUTPUT_SIZE = output_size

    json_files = sorted(scene_graph_dir.glob("*.json"))
    if max_files is not None:
        json_files = json_files[: max(0, max_files)]

    logger.info("Found %d scene-graph JSON files", len(json_files))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()

    total_rows = 0
    processed = 0
    failed = 0

    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCHEMA)
        writer.writeheader()

        for json_file in json_files:
            try:
                rows = process_mimic_json_rows(str(json_file))
                if rows:
                    writer.writerows(rows)
                    total_rows += len(rows)
                processed += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.warning("Skip %s due to error: %s", json_file.name, exc)

            if processed % 5000 == 0 and processed > 0:
                logger.info("Progress: processed=%d failed=%d", processed, failed)

    logger.info("Wrote %d rows to %s", total_rows, output_csv)
    logger.info("Processed files: %d | Failed files: %d", processed, failed)
    return pd.DataFrame(columns=SCHEMA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mimic_scene_graph.csv from ImaGenome JSON files")
    parser.add_argument("--scene-graph-dir", type=Path, default=DEFAULT_SCENE_GRAPH_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE, help="Target letterboxed size")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit for smoke tests")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger()

    process_scene_graph_directory(
        scene_graph_dir=args.scene_graph_dir,
        output_csv=args.output_csv,
        metadata_csv=args.metadata_csv,
        output_size=args.output_size,
        max_files=args.max_files,
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
