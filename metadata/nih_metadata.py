from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NIH_LABEL_COLUMNS = [
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

IMAGE_INDEX_RE = re.compile(r"^(?P<patient>\d+)_(?P<study>\d+)\.[A-Za-z0-9]+$")
NIH_IMAGE_ID_RE = re.compile(r"^NIH_(?P<patient>\d+)_(?P<study>\d+)_", re.IGNORECASE)


@dataclass(frozen=True)
class NihMetadataRecord:
    image_id: str
    image_path: str
    dataset: str
    labels: list[int]
    bboxes: list[dict[str, Any]]
    findings: str
    report: str
    patient_id: str
    study_id: str

    def to_json(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "dataset": self.dataset,
            "labels": self.labels,
            "bboxes": self.bboxes,
            "findings": self.findings,
            "report": self.report,
            "patient_id": self.patient_id,
            "study_id": self.study_id,
        }


def build_nih_metadata(
    metadata_json: Path,
    labels_csv: Path,
    output_dir: Path,
    *,
    output_filename: str = "nih_metadata.jsonl",
    logger: logging.Logger | None = None,
) -> list[NihMetadataRecord]:
    metadata_json = metadata_json.resolve()
    labels_csv = labels_csv.resolve()
    output_dir = output_dir.resolve()

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Loading NIH labels from %s", labels_csv)
    label_index = load_nih_label_index(labels_csv)
    logger.info("Loaded %d NIH label rows", len(label_index))

    logger.info("Loading NIH metadata from %s", metadata_json)
    metadata_rows = load_json_or_jsonl(metadata_json)
    logger.info("Loaded %d NIH metadata rows", len(metadata_rows))

    records: list[NihMetadataRecord] = []
    missing_labels = 0

    for row in metadata_rows:
        if not isinstance(row, dict):
            continue

        image_id = normalize_text(row.get("image_id"))
        image_path = normalize_text(row.get("image_path"))
        patient_id = normalize_text(row.get("patient_id"))
        study_id = normalize_text(row.get("study_id"))

        if not patient_id or not study_id:
            patient_id, study_id = extract_patient_study_from_image_id(image_id)

        if not image_id or not image_path or not patient_id or not study_id:
            continue

        labels = label_index.get((patient_id, study_id))
        if labels is None:
            missing_labels += 1
            continue

        records.append(
            NihMetadataRecord(
                image_id=image_id,
                image_path=image_path,
                dataset="nih",
                labels=labels,
                bboxes=[],
                findings="",
                report=compose_report("", ""),
                patient_id=patient_id,
                study_id=study_id,
            )
        )

    logger.info("Built %d NIH metadata records", len(records))
    logger.info("Skipped samples with missing labels: %d", missing_labels)
    write_nih_outputs(output_dir, output_filename, records)
    return records


def load_nih_label_index(labels_csv: Path) -> dict[tuple[str, str], list[int]]:
    label_index: dict[tuple[str, str], list[int]] = {}

    with labels_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            image_index = normalize_text(row.get("Image Index"))
            if not image_index:
                continue

            patient_id, study_id = extract_patient_study_from_image_index(image_index)
            if not patient_id or not study_id:
                continue

            findings = normalize_text(row.get("Finding Labels"))
            labels = findings_to_vector(findings)
            label_index[(patient_id, study_id)] = labels

    return label_index


def findings_to_vector(findings: str) -> list[int]:
    if not findings or findings.lower() == "no finding":
        return [0] * len(NIH_LABEL_COLUMNS)

    active_labels = {
        normalize_label_token(token)
        for token in findings.split("|")
        if normalize_label_token(token)
    }

    vector: list[int] = []
    for label_name in NIH_LABEL_COLUMNS:
        vector.append(1 if normalize_label_token(label_name) in active_labels else 0)
    return vector


def normalize_label_token(label: str) -> str:
    token = normalize_text(label).lower()
    token = token.replace("_", " ")
    token = re.sub(r"\s+", " ", token)
    return token


def extract_patient_study_from_image_index(image_index: str) -> tuple[str, str]:
    match = IMAGE_INDEX_RE.match(image_index)
    if not match:
        return "", ""

    patient = match.group("patient").zfill(8)
    study = match.group("study").zfill(3)
    return patient, study


def extract_patient_study_from_image_id(image_id: str) -> tuple[str, str]:
    match = NIH_IMAGE_ID_RE.match(image_id)
    if not match:
        return "", ""

    patient = match.group("patient").zfill(8)
    study = match.group("study").zfill(3)
    return patient, study


def write_nih_outputs(output_dir: Path, output_filename: str, records: list[NihMetadataRecord]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / output_filename
    label_map_path = output_dir / "nih_label_map.json"
    summary_path = output_dir / "nih_metadata_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_json(), ensure_ascii=False))
            stream.write("\n")

    label_map = {str(index): label_name for index, label_name in enumerate(NIH_LABEL_COLUMNS)}
    label_map_path.write_text(json.dumps(label_map, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "dataset": "nih",
        "record_count": len(records),
        "label_count": len(NIH_LABEL_COLUMNS),
        "label_columns": NIH_LABEL_COLUMNS,
        "jsonl_path": str(jsonl_path),
        "label_map_path": str(label_map_path),
        "sample_image_ids": [record.image_id for record in records[:5]],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json_or_jsonl(path: Path) -> list[Any]:
    raw_text = path.read_text(encoding="utf-8")
    stripped = raw_text.lstrip()
    if not stripped:
        return []

    if stripped[0] in "[{":
        try:
            payload = json.loads(raw_text)
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                return [payload]
        except json.JSONDecodeError:
            pass

    rows: list[Any] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def normalize_text(value: Any | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def compose_report(findings: str, impression: str) -> str:
    return f"FINDINGS: {findings}; IMPRESSION: {impression};"
