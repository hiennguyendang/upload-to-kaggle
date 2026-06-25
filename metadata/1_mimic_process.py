from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

CHECKPOINT_EVERY = 10000

DEFAULT_DIMENSIONS = Path(r"C:\Users\dhint\CHEX-DATA\MyChex\data\dimensions.jsonl")
DEFAULT_ENRICHED = Path(r"C:\Users\dhint\CHEX-DATA\MyChex\data\mimic_metadata_enriched.jsonl")
DEFAULT_REPORT_ROOT = Path(r"C:\Users\dhint\CHEX-DATA\MIMIC-CXR\REPORT__MIMIC")
DEFAULT_IMAGES_ROOT = Path(r"C:\Users\dhint\CHEX-DATA\MIMIC-CXR\images")
DEFAULT_SILVER_SCENE = Path(r"C:\Users\dhint\CHEX-DATA\MIMIC-CXR\ImaGenome\silver_dataset\scene_graph")
DEFAULT_GOLD_SCENE = Path(r"C:\Users\dhint\CHEX-DATA\MIMIC-CXR\ImaGenome\gold_dataset\scene_graph")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\dhint\CHEX-DATA\MIMIC-CXR\metadata")
DEFAULT_COPY_DIR = Path(r"C:\Users\dhint\CHEX-DATA\MyChex\data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MIMIC final metadata JSONL")
    parser.add_argument("--dimensions", type=Path, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--enriched", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--silver-scene", type=Path, default=DEFAULT_SILVER_SCENE)
    parser.add_argument("--gold-scene", type=Path, default=DEFAULT_GOLD_SCENE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--copy-dir", type=Path, default=DEFAULT_COPY_DIR)
    parser.add_argument("--output-name", type=str, default="mimic_metadata_final.jsonl")
    parser.add_argument("--reset", action="store_true", help="Delete checkpoint and raw output")
    return parser.parse_args()


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def count_lines(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as stream:
        for _ in stream:
            total += 1
    return total


def render_progress(current: int, total: int, prefix: str = "Progress") -> None:
    if total <= 0:
        return

    width = 28
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    message = f"\r{prefix}: [{bar}] {current:,}/{total:,} ({ratio * 100:5.1f}%)"
    sys.stdout.write(message)
    sys.stdout.flush()


def finish_progress(prefix: str = "Progress") -> None:
    sys.stdout.write("\n")
    sys.stdout.flush()


def load_enriched_index(enriched_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with enriched_path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            patient_id = str(row.get("patient_id", "")).strip()
            study_id = str(row.get("study_id", "")).strip()
            split = row.get("split")
            labels = row.get("labels")
            if not patient_id or not study_id:
                continue
            if not split or not isinstance(labels, list):
                continue

            key = (patient_id, study_id)
            if key not in index:
                index[key] = {"split": str(split), "labels": labels}

    return index


def extract_dicom_id(image_id: str) -> str:
    if not image_id:
        return ""
    parts = image_id.split("_")
    if len(parts) < 4:
        return ""
    return "_".join(parts[3:])


def build_image_path(images_root: Path, patient_id: str, image_id: str) -> str:
    prefix = patient_id[:2].zfill(2)
    return str((images_root / f"p{prefix}" / f"p{patient_id}" / f"{image_id}.jpg").resolve())


def build_report_path(report_root: Path, patient_id: str, study_id: str) -> Path:
    prefix = patient_id[:2].zfill(2)
    return report_root / f"p{prefix}" / f"p{patient_id}" / f"s{study_id}.txt"


def load_report(report_root: Path, patient_id: str, study_id: str, cache: Dict[Tuple[str, str], str]) -> str:
    key = (patient_id, study_id)
    if key in cache:
        return cache[key]

    report_path = build_report_path(report_root, patient_id, study_id)
    if not report_path.exists():
        cache[key] = ""
        return ""

    text = report_path.read_text(encoding="utf-8", errors="ignore").strip()
    cache[key] = text
    return text


def build_scene_path(
    split: str,
    image_id: str,
    silver_dir: Path,
    gold_dir: Path,
) -> str:
    dicom_id = extract_dicom_id(image_id)
    if not dicom_id:
        return ""

    scene_dir = gold_dir if split == "gold" else silver_dir
    scene_path = scene_dir / f"{dicom_id}_SceneGraph.json"
    return str(scene_path) if scene_path.exists() else ""


def make_record(
    patient_id: str,
    study_id: str,
    image_id: str,
    split: str,
    image_path: str,
    scene_path: str,
    labels: list[int],
    report: str,
    width: int,
    height: int,
) -> Dict[str, Any]:
    return {
        "patient_id": patient_id,
        "study_id": study_id,
        "image_id": image_id,
        "dataset": "mimic",
        "split": split,
        "image_path": image_path,
        "scene_path": scene_path,
        "labels": labels,
        "report": report,
        "width": width,
        "height": height,
    }


def build_raw_records(
    dimensions_path: Path,
    enriched_index: Dict[Tuple[str, str], Dict[str, Any]],
    report_root: Path,
    images_root: Path,
    silver_scene: Path,
    gold_scene: Path,
    output_dir: Path,
    raw_path: Path,
    checkpoint_path: Path,
    reset: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    total_lines = count_lines(dimensions_path)

    if reset:
        if raw_path.exists():
            raw_path.unlink()
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    checkpoint = load_checkpoint(checkpoint_path)
    lines_read = int(checkpoint.get("lines_read", 0))
    raw_complete = bool(checkpoint.get("raw_complete", False))

    if raw_complete and raw_path.exists():
        return raw_path

    if lines_read > 0 and not raw_path.exists():
        lines_read = 0

    report_cache: Dict[Tuple[str, str], str] = {}
    mode = "a" if lines_read > 0 else "w"
    processed_since_refresh = 0

    with dimensions_path.open("r", encoding="utf-8") as dims, raw_path.open(mode, encoding="utf-8") as out:
        for line_no, raw_line in enumerate(dims):
            if line_no < lines_read:
                continue

            lines_read = line_no + 1
            processed_since_refresh += 1
            line = raw_line.strip()
            if not line:
                if lines_read % CHECKPOINT_EVERY == 0:
                    write_checkpoint(checkpoint_path, {"lines_read": lines_read, "raw_complete": False})
                if processed_since_refresh >= 1000:
                    render_progress(lines_read, total_lines, prefix="MIMIC metadata")
                    processed_since_refresh = 0
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if lines_read % CHECKPOINT_EVERY == 0:
                    write_checkpoint(checkpoint_path, {"lines_read": lines_read, "raw_complete": False})
                if processed_since_refresh >= 1000:
                    render_progress(lines_read, total_lines, prefix="MIMIC metadata")
                    processed_since_refresh = 0
                continue

            patient_id = str(row.get("patient_id", "")).strip()
            study_id = str(row.get("study_id", "")).strip()
            image_id = str(row.get("image_id", "")).strip()
            width = row.get("width")
            height = row.get("height")
            scene_path_from_dimensions = str(row.get("scene_path", "")).strip()
            if not patient_id or not study_id or not image_id:
                if lines_read % CHECKPOINT_EVERY == 0:
                    write_checkpoint(checkpoint_path, {"lines_read": lines_read, "raw_complete": False})
                if processed_since_refresh >= 1000:
                    render_progress(lines_read, total_lines, prefix="MIMIC metadata")
                    processed_since_refresh = 0
                continue

            join = enriched_index.get((patient_id, study_id))
            if join is None:
                if lines_read % CHECKPOINT_EVERY == 0:
                    write_checkpoint(checkpoint_path, {"lines_read": lines_read, "raw_complete": False})
                if processed_since_refresh >= 1000:
                    render_progress(lines_read, total_lines, prefix="MIMIC metadata")
                    processed_since_refresh = 0
                continue

            split = join.get("split")
            labels = join.get("labels")
            if not split or not isinstance(labels, list):
                if lines_read % CHECKPOINT_EVERY == 0:
                    write_checkpoint(checkpoint_path, {"lines_read": lines_read, "raw_complete": False})
                if processed_since_refresh >= 1000:
                    render_progress(lines_read, total_lines, prefix="MIMIC metadata")
                    processed_since_refresh = 0
                continue

            report = load_report(report_root, patient_id, study_id, report_cache)
            if not report:
                if lines_read % CHECKPOINT_EVERY == 0:
                    write_checkpoint(checkpoint_path, {"lines_read": lines_read, "raw_complete": False})
                if processed_since_refresh >= 1000:
                    render_progress(lines_read, total_lines, prefix="MIMIC metadata")
                    processed_since_refresh = 0
                continue

            image_path = build_image_path(images_root, patient_id, image_id)
            if scene_path_from_dimensions:
                scene_path = scene_path_from_dimensions
            else:
                scene_path = build_scene_path(split, image_id, silver_scene, gold_scene)

            record = make_record(
                patient_id=patient_id,
                study_id=study_id,
                image_id=image_id,
                split=str(split),
                image_path=image_path,
                scene_path=scene_path,
                labels=labels,
                report=report,
                width=int(width) if isinstance(width, (int, float)) else width,
                height=int(height) if isinstance(height, (int, float)) else height,
            )
            out.write(json.dumps(record))
            out.write("\n")

            if lines_read % CHECKPOINT_EVERY == 0:
                write_checkpoint(checkpoint_path, {"lines_read": lines_read, "raw_complete": False})

            if processed_since_refresh >= 1000:
                render_progress(lines_read, total_lines, prefix="MIMIC metadata")
                processed_since_refresh = 0

    write_checkpoint(checkpoint_path, {"lines_read": lines_read, "raw_complete": True})
    render_progress(total_lines, total_lines, prefix="MIMIC metadata")
    finish_progress(prefix="MIMIC metadata")
    return raw_path


def select_best_record(
    current: Tuple[Dict[str, Any], str] | None,
    candidate: Dict[str, Any],
    candidate_dicom: str,
) -> Tuple[Dict[str, Any], str]:
    if current is None:
        return candidate, candidate_dicom

    current_row, current_dicom = current
    current_scene = bool(current_row.get("scene_path"))
    candidate_scene = bool(candidate.get("scene_path"))

    if not current_scene and candidate_scene:
        return candidate, candidate_dicom

    if current_scene == candidate_scene and candidate_dicom > current_dicom:
        return candidate, candidate_dicom

    return current_row, current_dicom


def dedupe_records(raw_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    best: Dict[Tuple[str, str], Tuple[Dict[str, Any], str]] = {}

    with raw_path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            patient_id = str(row.get("patient_id", "")).strip()
            study_id = str(row.get("study_id", "")).strip()
            image_id = str(row.get("image_id", "")).strip()
            if not patient_id or not study_id or not image_id:
                continue

            dicom_id = extract_dicom_id(image_id)
            key = (patient_id, study_id)
            current = best.get(key)
            best[key] = select_best_record(current, row, dicom_id)

    return {key: value[0] for key, value in best.items()}


def normalize_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "patient_id": row.get("patient_id", ""),
        "study_id": row.get("study_id", ""),
        "image_id": row.get("image_id", ""),
        "dataset": row.get("dataset", "mimic"),
        "split": row.get("split", ""),
        "image_path": row.get("image_path", ""),
        "scene_path": row.get("scene_path", ""),
        "labels": row.get("labels", []),
        "report": row.get("report", ""),
        "width": row.get("width"),
        "height": row.get("height"),
    }


def write_final(records: Dict[Tuple[str, str], Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in records.values():
            stream.write(json.dumps(normalize_record(row)))
            stream.write("\n")


def build_summary(
    records: Dict[Tuple[str, str], Dict[str, Any]],
    gold_scene_dir: Path,
    output_path: Path,
) -> Dict[str, Any]:
    total_lines = len(records)
    patient_ids = {row.get("patient_id", "") for row in records.values() if row.get("patient_id")}
    scene_path_count = sum(1 for row in records.values() if row.get("scene_path"))
    labels_count = sum(1 for row in records.values() if isinstance(row.get("labels"), list) and row.get("labels"))
    report_count = sum(1 for row in records.values() if row.get("report"))
    gold_split_count = sum(1 for row in records.values() if row.get("split") == "gold")
    gold_split_scene_count = sum(
        1 for row in records.values() if row.get("split") == "gold" and row.get("scene_path")
    )

    expected_gold_files = set()
    for row in records.values():
        if row.get("split") != "gold":
            continue
        dicom_id = extract_dicom_id(str(row.get("image_id", "")))
        if dicom_id:
            expected_gold_files.add(f"{dicom_id}_SceneGraph.json")

    actual_gold_files = set()
    if gold_scene_dir.exists():
        for scene_file in gold_scene_dir.glob("*.json"):
            if scene_file.is_file():
                actual_gold_files.add(scene_file.name)

    unmatched_gold_files = sorted(actual_gold_files - expected_gold_files)

    return {
        "dataset": "mimic",
        "record_count": total_lines,
        "patient_count": len(patient_ids),
        "scene_path_count": scene_path_count,
        "labels_count": labels_count,
        "report_count": report_count,
        "gold_split_count": gold_split_count,
        "gold_split_scene_count": gold_split_scene_count,
        "gold_scene_unmatched_files": unmatched_gold_files,
        "output_path": str(output_path),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_path = output_dir / args.output_name
    raw_path = output_dir / "mimic_metadata_final_raw.jsonl"
    checkpoint_path = output_dir / "mimic_metadata_final_checkpoint.json"
    summary_path = output_dir / "mimic_metadata_final_summary.json"

    enriched_index = load_enriched_index(args.enriched.resolve())

    build_raw_records(
        dimensions_path=args.dimensions.resolve(),
        enriched_index=enriched_index,
        report_root=args.report_root.resolve(),
        images_root=args.images_root.resolve(),
        silver_scene=args.silver_scene.resolve(),
        gold_scene=args.gold_scene.resolve(),
        output_dir=output_dir,
        raw_path=raw_path,
        checkpoint_path=checkpoint_path,
        reset=args.reset,
    )

    records = dedupe_records(raw_path)
    write_final(records, output_path)

    summary = build_summary(records, args.gold_scene.resolve(), output_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    args.copy_dir.resolve().mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_path, args.copy_dir.resolve() / output_path.name)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
