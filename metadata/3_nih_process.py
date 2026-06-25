from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

CHECKPOINT_EVERY = 10000

DEFAULT_DIMENSIONS = Path(r"C:\Users\dhint\CHEX-DATA\NIH\dimensions.jsonl")
DEFAULT_SPLITS = Path(r"C:\Users\dhint\CHEX-DATA\NIH\metadata\nih_splits.csv")
DEFAULT_LABELS = Path(r"C:\Users\dhint\CHEX-DATA\NIH\Data_Entry_2017.csv")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\dhint\CHEX-DATA\NIH\metadata")
DEFAULT_COPY_DIR = Path(r"C:\Users\dhint\CHEX-DATA\MyChex\data")

CHEXPERT_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

NIH_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural Thickening",
    "Hernia",
]

NIH_TO_CHEX_MAP = {
    0: 0,
    1: 1,
    2: 9,
    3: 7,
    4: 6,
    5: 6,
    6: 11,
    7: 12,
    8: 2,
    9: 3,
    12: 10,
}

NORMAL_REPORT = (
    "The chest X-ray is normal. "
    "No significant findings or acute cardiopulmonary abnormalities are seen."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NIH final metadata JSONL")
    parser.add_argument("--dimensions", type=Path, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--copy-dir", type=Path, default=DEFAULT_COPY_DIR)
    parser.add_argument("--output-name", type=str, default="nih_metadata_final.jsonl")
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


def normalize_patient_id(value: str) -> str:
    cleaned = value.strip()
    if cleaned.isdigit():
        return cleaned.zfill(8)
    return cleaned


def normalize_study_id(value: str) -> str:
    cleaned = value.strip()
    if cleaned.isdigit():
        return cleaned.zfill(3)
    return cleaned


def parse_image_index(image_index: str) -> Tuple[str, str]:
    stem = Path(image_index).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return "", ""
    patient_id = normalize_patient_id(parts[0])
    study_id = normalize_study_id(parts[1])
    return patient_id, study_id


def normalize_label_token(label: str) -> str:
    token = label.strip().lower().replace("_", " ")
    return " ".join(token.split())


def map_findings_to_labels(findings: str) -> list[int]:
    if not findings or findings.lower() == "no finding":
        findings_list = []
    else:
        findings_list = [normalize_label_token(token) for token in findings.split("|")]

    nih_index = {normalize_label_token(label): idx for idx, label in enumerate(NIH_LABELS)}
    positive_indices = {nih_index[token] for token in findings_list if token in nih_index}

    new_labels = [-100] * len(CHEXPERT_LABELS)
    has_positive = False
    for idx in positive_indices:
        if idx in NIH_TO_CHEX_MAP:
            chex_idx = NIH_TO_CHEX_MAP[idx]
            new_labels[chex_idx] = 1
            has_positive = True

    if not has_positive:
        new_labels[8] = 1

    return new_labels


def labels_to_report(labels: list[int]) -> str:
    findings = [CHEXPERT_LABELS[i] for i, val in enumerate(labels) if val == 1]

    if not findings:
        return compose_report("", NORMAL_REPORT)

    if len(findings) == 1:
        impression_text = f"The chest X-ray shows evidence of {findings[0].lower()}."
    elif len(findings) == 2:
        impression_text = f"The chest X-ray shows evidence of {findings[0].lower()} and {findings[1].lower()}."
    else:
        finding_list = ", ".join([f.lower() for f in findings[:-1]]) + f", and {findings[-1].lower()}"
        impression_text = f"The chest X-ray shows evidence of {finding_list}."

    return compose_report("", impression_text)


def compose_report(findings: str, impression: str) -> str:
    return f"FINDINGS: {findings}; IMPRESSION: {impression};"


def load_split_index(splits_path: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    with splits_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            patient_id = normalize_patient_id(str(row.get("patient_id", "")))
            split = str(row.get("split", "")).strip()
            if patient_id and split and patient_id not in index:
                index[patient_id] = split
    return index


def load_label_index(labels_path: Path) -> Dict[Tuple[str, str], list[int]]:
    index: Dict[Tuple[str, str], list[int]] = {}
    with labels_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            image_index = str(row.get("Image Index", "")).strip()
            if not image_index:
                continue
            patient_id, study_id = parse_image_index(image_index)
            if not patient_id or not study_id:
                continue

            findings = str(row.get("Finding Labels", "")).strip()
            labels = map_findings_to_labels(findings)
            key = (patient_id, study_id)
            if key not in index:
                index[key] = labels
    return index


def extract_dicom_id(image_id: str) -> str:
    if not image_id:
        return ""
    parts = image_id.split("_")
    if len(parts) < 4:
        return ""
    return "_".join(parts[3:])


def build_image_path(image_id: str) -> str:
    return str((Path(r"C:\Users\dhint\CHEX-DATA\NIH\images") / f"{image_id}.png").resolve())


def make_record(
    patient_id: str,
    study_id: str,
    image_id: str,
    split: str,
    image_path: str,
    labels: list[int],
    report: str,
    width: int,
    height: int,
) -> Dict[str, Any]:
    return {
        "patient_id": patient_id,
        "study_id": study_id,
        "image_id": image_id,
        "dataset": "nih",
        "split": split,
        "image_path": image_path,
        "scene_path": "",
        "labels": labels,
        "report": report,
        "width": width,
        "height": height,
    }


def build_raw_records(
    dimensions_path: Path,
    split_index: Dict[str, str],
    label_index: Dict[Tuple[str, str], list[int]],
    output_dir: Path,
    raw_path: Path,
    checkpoint_path: Path,
    reset: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

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

    mode = "a" if lines_read > 0 else "w"

    with dimensions_path.open("r", encoding="utf-8") as dims, raw_path.open(mode, encoding="utf-8") as out:
        for line_no, raw_line in enumerate(dims):
            if line_no < lines_read:
                continue

            lines_read = line_no + 1
            line = raw_line.strip()
            if not line:
                if lines_read % CHECKPOINT_EVERY == 0:
                    write_checkpoint(checkpoint_path, {"lines_read": lines_read, "raw_complete": False})
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if lines_read % CHECKPOINT_EVERY == 0:
                    write_checkpoint(checkpoint_path, {"lines_read": lines_read, "raw_complete": False})
                continue

            patient_id = normalize_patient_id(str(row.get("patient_id", "")))
            study_id = normalize_study_id(str(row.get("study_id", "")))
            image_id = str(row.get("image_id", "")).strip()
            width = row.get("width")
            height = row.get("height")
            if not patient_id or not study_id or not image_id:
                if lines_read % CHECKPOINT_EVERY == 0:
                    write_checkpoint(checkpoint_path, {"lines_read": lines_read, "raw_complete": False})
                continue

            split = split_index.get(patient_id)
            if not split:
                if lines_read % CHECKPOINT_EVERY == 0:
                    write_checkpoint(checkpoint_path, {"lines_read": lines_read, "raw_complete": False})
                continue

            labels = label_index.get((patient_id, study_id))
            if labels is None:
                if lines_read % CHECKPOINT_EVERY == 0:
                    write_checkpoint(checkpoint_path, {"lines_read": lines_read, "raw_complete": False})
                continue

            report = labels_to_report(labels)
            if not report:
                if lines_read % CHECKPOINT_EVERY == 0:
                    write_checkpoint(checkpoint_path, {"lines_read": lines_read, "raw_complete": False})
                continue

            image_path = build_image_path(image_id)
            record = make_record(
                patient_id=patient_id,
                study_id=study_id,
                image_id=image_id,
                split=split,
                image_path=image_path,
                labels=labels,
                report=report,
                width=int(width) if isinstance(width, (int, float)) else width,
                height=int(height) if isinstance(height, (int, float)) else height,
            )
            out.write(json.dumps(record))
            out.write("\n")

            if lines_read % CHECKPOINT_EVERY == 0:
                write_checkpoint(checkpoint_path, {"lines_read": lines_read, "raw_complete": False})

    write_checkpoint(checkpoint_path, {"lines_read": lines_read, "raw_complete": True})
    return raw_path


def select_best_record(
    current: Tuple[Dict[str, Any], str] | None,
    candidate: Dict[str, Any],
    candidate_dicom: str,
) -> Tuple[Dict[str, Any], str]:
    if current is None:
        return candidate, candidate_dicom

    current_row, current_dicom = current
    if candidate_dicom > current_dicom:
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
        "dataset": row.get("dataset", "nih"),
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


def build_summary(records: Dict[Tuple[str, str], Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
    total_lines = len(records)
    patient_ids = {row.get("patient_id", "") for row in records.values() if row.get("patient_id")}
    labels_count = sum(1 for row in records.values() if isinstance(row.get("labels"), list) and row.get("labels"))
    report_count = sum(1 for row in records.values() if row.get("report"))

    return {
        "dataset": "nih",
        "record_count": total_lines,
        "patient_count": len(patient_ids),
        "labels_count": labels_count,
        "report_count": report_count,
        "output_path": str(output_path),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_path = output_dir / args.output_name
    raw_path = output_dir / "nih_metadata_final_raw.jsonl"
    checkpoint_path = output_dir / "nih_metadata_final_checkpoint.json"
    summary_path = output_dir / "nih_metadata_final_summary.json"

    split_index = load_split_index(args.splits.resolve())
    label_index = load_label_index(args.labels.resolve())

    build_raw_records(
        dimensions_path=args.dimensions.resolve(),
        split_index=split_index,
        label_index=label_index,
        output_dir=output_dir,
        raw_path=raw_path,
        checkpoint_path=checkpoint_path,
        reset=args.reset,
    )

    records = dedupe_records(raw_path)
    write_final(records, output_path)

    summary = build_summary(records, output_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    args.copy_dir.resolve().mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_path, args.copy_dir.resolve() / output_path.name)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
