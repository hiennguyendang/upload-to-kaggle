from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
LABEL_VALUE_RE = re.compile(r"^-?\d+(?:\.0+)?$")
PATIENT_FOLDER_RE = re.compile(r"^p\d+$", re.IGNORECASE)
STUDY_ID_RE = re.compile(r"(?:^|[_/])(?:s|study)(\d+)", re.IGNORECASE)
MIMIC_IMAGE_NAME_RE = re.compile(
    r"^MIMIC_(?P<patient>p\d+)_(?P<study>s\d+)_(?P<digest>[a-f0-9]+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MimicMetadataRecord:
    image_id: str
    image_path: str
    dataset: str
    labels: list[int]
    report: str
    patient_id: str
    study_id: str
    study_time: str

    def to_json(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "dataset": self.dataset,
            "labels": self.labels,
            "report": self.report,
            "patient_id": self.patient_id,
            "study_id": self.study_id,
            "study_time": self.study_time,
        }


def build_mimic_metadata(
    dataset_root: Path,
    output_dir: Path | None = None,
    *,
    images_root: Path | None = None,
    report_root: Path | None = None,
    labels_csv: Path | None = None,
    metadata_csv: Path | None = None,
    logger: logging.Logger | None = None,
) -> list[MimicMetadataRecord]:
    dataset_root = dataset_root.resolve()
    output_dir = (output_dir or (dataset_root / "metadata")).resolve()
    images_root = (images_root or (dataset_root / "images")).resolve()
    report_root = (report_root or (dataset_root / "REPORT__MIMIC")).resolve()
    labels_csv = (labels_csv or (dataset_root / "mimic-cxr-2.0.0-chexpert.csv")).resolve()
    metadata_csv = (metadata_csv or (dataset_root / "mimic-cxr-2.0.0-metadata.csv")).resolve()

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Loading MIMIC labels from %s", labels_csv)
    label_index, label_columns = load_mimic_labels(labels_csv)
    logger.info("Loaded %d label rows", len(label_index))

    logger.info("Loading MIMIC study metadata from %s", metadata_csv)
    study_time_index = load_mimic_study_times(metadata_csv)
    logger.info("Loaded %d study-time rows", len(study_time_index))

    logger.info("Loading MIMIC reports from %s", report_root)
    report_index = load_mimic_reports(report_root)
    logger.info("Loaded %d report rows", len(report_index))

    logger.info("Scanning MIMIC image files in %s", images_root)
    image_paths = sorted(
        [
            image_path
            for image_path in images_root.rglob("*")
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]
    )
    logger.info("Found %d candidate image files", len(image_paths))

    records: list[MimicMetadataRecord] = []
    missing_labels = 0
    missing_reports = 0
    empty_report_sections = 0

    for image_path in image_paths:
        raw_patient_id, raw_study_id = extract_patient_and_study(image_path)
        patient_id = normalize_subject_id(raw_patient_id)
        study_id = normalize_study_id(raw_study_id)
        image_id = extract_image_id(image_path, raw_patient_id, raw_study_id)
        labels = label_index.get((patient_id, study_id))
        report_entry = report_index.get((patient_id, study_id))
        if report_entry is None:
            findings, impression = "", ""
            missing_reports += 1
        else:
            findings, impression = report_entry
        report = compose_standard_report(findings, impression)
        study_time = study_time_index.get((patient_id, study_id), "")

        if labels is None:
            missing_labels += 1
            continue

        if report_entry is not None and not findings and not impression:
            empty_report_sections += 1

        records.append(
            MimicMetadataRecord(
                image_id=image_id,
                image_path=str(image_path.resolve()),
                dataset="mimic",
                labels=labels,
                report=report,
                patient_id=patient_id,
                study_id=study_id,
                study_time=study_time,
            )
        )

    logger.info("Built %d metadata records", len(records))
    logger.info("Missing labels: %d", missing_labels)
    logger.info("Missing reports: %d", missing_reports)
    logger.info("Empty report sections: %d", empty_report_sections)
    write_mimic_outputs(output_dir, records, label_columns)
    return records


def load_mimic_labels(labels_csv: Path) -> tuple[dict[tuple[str, str], list[int]], list[str]]:
    label_index: dict[tuple[str, str], list[int]] = {}

    with labels_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        label_columns = [column for column in columns if column not in {"subject_id", "study_id"}]

        for row in reader:
            subject_id = normalize_subject_id(normalize_id(row.get("subject_id")))
            study_id = normalize_study_id(normalize_id(row.get("study_id")))
            if not subject_id or not study_id:
                continue

            labels = [normalize_label_value(row.get(column)) for column in label_columns]
            label_index[(subject_id, study_id)] = labels

    return label_index, label_columns


def load_mimic_study_times(metadata_csv: Path) -> dict[tuple[str, str], str]:
    study_time_index: dict[tuple[str, str], str] = {}

    with metadata_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            subject_id = normalize_subject_id(normalize_id(row.get("subject_id")))
            study_id = normalize_study_id(normalize_id(row.get("study_id")))
            if not subject_id or not study_id:
                continue

            study_time = normalize_study_time(row.get("StudyTime"))
            if study_time and (subject_id, study_id) not in study_time_index:
                study_time_index[(subject_id, study_id)] = study_time

    return study_time_index


def load_mimic_reports(report_root: Path) -> dict[tuple[str, str], tuple[str, str]]:
    report_index: dict[tuple[str, str], tuple[str, str]] = {}

    for report_path in sorted(report_root.rglob("*.txt")):
        patient_id, study_id = extract_patient_and_study(report_path)
        patient_id = normalize_subject_id(patient_id)
        study_id = normalize_study_id(study_id)
        raw_text = report_path.read_text(encoding="utf-8", errors="ignore")
        findings, impression = extract_findings_impression(raw_text)
        report_index[(patient_id, study_id)] = (findings, impression)

    return report_index


def write_mimic_outputs(output_dir: Path, records: list[MimicMetadataRecord], label_columns: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "mimic_metadata.jsonl"
    label_map_path = output_dir / "mimic_label_map.json"
    summary_path = output_dir / "mimic_metadata_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_json(), ensure_ascii=False))
            stream.write("\n")

    label_map = {str(index): label_name for index, label_name in enumerate(label_columns)}
    label_map_path.write_text(json.dumps(label_map, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "dataset": "mimic",
        "record_count": len(records),
        "label_count": len(label_columns),
        "label_columns": label_columns,
        "jsonl_path": str(jsonl_path),
        "label_map_path": str(label_map_path),
        "sample_image_ids": [record.image_id for record in records[:5]],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_patient_and_study(path: Path) -> tuple[str, str]:
    if PATIENT_FOLDER_RE.fullmatch(path.parent.name):
        patient_id = path.parent.name
    else:
        patient_id = "unknown_patient"

    stem = path.stem
    name_match = MIMIC_IMAGE_NAME_RE.match(stem)
    if name_match:
        return name_match.group("patient"), name_match.group("study")

    study_match = STUDY_ID_RE.search(stem)
    if study_match:
        return patient_id, f"s{study_match.group(1)}"

    for part in reversed(path.parts):
        if part.lower().startswith("s") and part[1:].isdigit():
            return patient_id, part.lower()

    return patient_id, "unknown_study"


def extract_image_id(path: Path, patient_id: str, study_id: str) -> str:
    stem = path.stem
    if stem:
        return stem

    unique_key = f"{patient_id}/{study_id}/{path.name}".replace("\\", "/").lower()
    digest = hashlib.sha1(unique_key.encode("utf-8")).hexdigest()[:12]
    return f"MIMIC_{patient_id}_{study_id}_{digest}"


def normalize_id(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = value.strip()
    return cleaned


def normalize_subject_id(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned.startswith("p") and cleaned[1:].isdigit():
        return cleaned[1:]
    return cleaned


def normalize_study_id(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned.startswith("s") and cleaned[1:].isdigit():
        return cleaned[1:]
    return cleaned


def normalize_label_value(value: str | None) -> int:
    if value is None:
        return -100

    cleaned = value.strip()
    if not cleaned:
        return -100

    if not LABEL_VALUE_RE.match(cleaned):
        try:
            numeric_value = int(float(cleaned))
        except ValueError:
            return -100
    else:
        numeric_value = int(float(cleaned))

    if numeric_value == -1:
        return -100
    return numeric_value if numeric_value in {0, 1} else -100


def normalize_study_time(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = value.strip()
    if not cleaned:
        return ""
    return cleaned.zfill(6) if cleaned.isdigit() else cleaned


def clean_report_text(raw_text: str) -> str:
    lines = [line.rstrip() for line in raw_text.splitlines()]
    non_empty_lines = [line for line in lines if line.strip()]
    return "\n".join(non_empty_lines).strip()


def extract_findings_impression(raw_report: str) -> tuple[str, str]:
    if not raw_report or not isinstance(raw_report, str):
        return "", ""

    text = raw_report.lower()
    findings_pattern = re.compile(r"findings:\s*(.*?)(?=(?:[a-z][a-z\s]*:)|$)", re.DOTALL)
    impression_pattern = re.compile(r"impression:\s*(.*?)(?=(?:[a-z][a-z\s]*:)|$)", re.DOTALL)

    findings_match = findings_pattern.search(text)
    impression_match = impression_pattern.search(text)

    findings = normalize_section_text(findings_match.group(1)) if findings_match else ""
    impression = normalize_section_text(impression_match.group(1)) if impression_match else ""

    if findings or impression:
        return findings, impression

    fallback = re.sub(
        r"(final report|examination:|indication:|technique:|comparison:).*?(?=(?:[a-z][a-z\s]*:)|$)",
        "",
        text,
        flags=re.DOTALL,
    )
    cleaned = normalize_section_text(fallback)
    return cleaned, ""


def normalize_section_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed


def compose_standard_report(findings: str, impression: str) -> str:
    return f"FINDINGS: {findings}; IMPRESSION: {impression};"
