from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PATIENT_TOKEN_RE = re.compile(r"patient\d+", re.IGNORECASE)
STUDY_TOKEN_RE = re.compile(r"study\d+", re.IGNORECASE)
VIEW_TOKEN_RE = re.compile(r"view\d+_(?:frontal|lateral)", re.IGNORECASE)


@dataclass(frozen=True)
class ChexplusMetadataRecord:
    image_id: str
    image_path: str
    dataset: str
    labels: list[int]
    report: str
    patient_id: str
    study_id: str
    view: str

    def to_json(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "dataset": self.dataset,
            "labels": self.labels,
            "report": self.report,
            "patient_id": self.patient_id,
            "study_id": self.study_id,
            "view": self.view,
        }


def build_chexplus_metadata(
    images_source: Path,
    csv_path: Path,
    labels_json: Path,
    output_dir: Path | None = None,
    *,
    processed_images_root: Path,
    output_filename: str = "chexplus_unified.jsonl",
    logger: logging.Logger | None = None,
) -> list[ChexplusMetadataRecord]:
    images_source = images_source.resolve()
    csv_path = csv_path.resolve()
    labels_json = labels_json.resolve()
    processed_images_root = processed_images_root.resolve()
    output_dir = (output_dir or images_source.parent).resolve()

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Loading CheXplus CSV from %s", csv_path)
    csv_index = load_chexplus_csv_index(csv_path)
    logger.info("Loaded %d CSV rows", len(csv_index))

    logger.info("Loading CheXplus labels from %s", labels_json)
    label_index = load_chexplus_labels(labels_json)
    logger.info("Loaded %d label rows", len(label_index))

    logger.info("Loading CheXplus image manifests from %s", images_source)
    image_paths = load_chexplus_image_paths(images_source, processed_images_root)
    logger.info("Loaded %d image rows", len(image_paths))

    records: list[ChexplusMetadataRecord] = []
    skipped_missing_csv = 0
    skipped_missing_labels = 0

    for image_path in image_paths:
        patient_id, study_id, view = extract_image_triplet(image_path)
        if not patient_id or not study_id or not view:
            continue

        key = (patient_id, study_id, view)
        csv_row = csv_index.get(key)
        if csv_row is None:
            skipped_missing_csv += 1
            continue

        findings = normalize_report_text(csv_row.get("findings"))
        impression = normalize_report_text(csv_row.get("impression"))
        report = compose_report(findings, impression)

        image_id = Path(image_path.replace("\\", "/")).stem
        labels = label_index.get(key)
        if labels is None:
            skipped_missing_labels += 1
            continue

        records.append(
            ChexplusMetadataRecord(
                image_id=image_id,
                image_path=image_path,
                dataset="chexplus",
                labels=labels,
                report=report,
                patient_id=patient_id,
                study_id=study_id,
                view=view,
            )
        )

    logger.info("Built %d metadata records", len(records))
    logger.info("Skipped samples with missing CSV key: %d", skipped_missing_csv)
    logger.info("Skipped samples with missing labels: %d", skipped_missing_labels)
    write_chexplus_outputs(output_dir, output_filename, records)
    return records


def load_chexplus_csv_index(csv_path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    csv_index: dict[tuple[str, str, str], dict[str, str]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            raw_path = normalize_source_path(row.get("path_to_image"))
            patient_id, study_id, view = extract_image_triplet(raw_path)
            if not patient_id or not study_id or not view:
                continue

            key = (patient_id, study_id, view)
            if key in csv_index:
                continue

            csv_index[key] = {
                "findings": normalize_report_text(row.get("section_findings")),
                "impression": normalize_report_text(row.get("section_impression")),
            }

    return csv_index


def load_chexplus_labels(labels_json: Path) -> dict[tuple[str, str, str], list[int]]:
    label_index: dict[tuple[str, str, str], list[int]] = {}
    payload = read_json_or_jsonl(labels_json)

    for row in iter_label_rows(payload):
        if not isinstance(row, dict):
            continue

        image_path = normalize_source_path(
            row.get("path_to_image")
            or row.get("image_path")
            or row.get("path")
            or row.get("file_path")
            or row.get("image")
            or row.get("filename")
        )
        if not image_path:
            continue

        patient_id, study_id, view = extract_image_triplet(image_path)
        if not patient_id or not study_id:
            continue

        labels = extract_label_vector(row)
        if labels is None or not labels:
            continue

        key = (patient_id, study_id, view)
        if key not in label_index:
            label_index[key] = labels

    return label_index


def load_chexplus_image_paths(images_source: Path, processed_images_root: Path) -> list[str]:
    if images_source.suffix.lower() in {".json", ".jsonl"}:
        payload = read_json_or_jsonl(images_source)
        return [
            path
            for path in collect_paths_from_payload(payload, processed_images_root)
            if path
        ]

    paths: list[str] = []
    for raw_line in images_source.read_text(encoding="utf-8").splitlines():
        path = normalize_image_path(raw_line, processed_images_root)
        if path:
            paths.append(path)
    return paths


def write_chexplus_outputs(output_dir: Path, output_filename: str, records: list[ChexplusMetadataRecord]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / output_filename
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_json(), ensure_ascii=False))
            stream.write("\n")


def extract_label_vector(row: dict[str, Any]) -> list[int] | None:
    label_fields = infer_label_fields(row)
    if not label_fields:
        return None

    return [normalize_label_value(row.get(field)) for field in label_fields]


def infer_label_fields(row: dict[str, Any]) -> list[str]:
    excluded_fields = {
        "path_to_image",
        "image_path",
        "path",
        "file_path",
        "image",
        "filename",
        "patient_id",
        "study_id",
        "view",
        "dataset",
        "image_id",
        "labels",
        "label",
        "text",
        "findings",
        "report",
        "finding",
    }
    return [key for key in row.keys() if key not in excluded_fields]


def iter_label_rows(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if isinstance(payload, dict):
        for key in ("data", "labels", "items", "entries", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]

        if any(key in payload for key in ("path_to_image", "image_path", "path", "labels", "label")):
            return [payload]

        return [row for row in payload.values() if isinstance(row, dict)]

    return []


def extract_image_triplet(path_text: str) -> tuple[str, str, str]:
    normalized_path = normalize_source_path(path_text)
    patient_id = extract_token(PATIENT_TOKEN_RE, normalized_path)
    study_id = extract_token(STUDY_TOKEN_RE, normalized_path)
    view = extract_token(VIEW_TOKEN_RE, normalized_path)
    return patient_id, study_id, view


def extract_token(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text.replace("\\", "/"))
    return match.group(0).lower() if match else ""


def collect_paths_from_payload(payload: Any, processed_images_root: Path) -> list[str]:
    if isinstance(payload, list):
        paths: list[str] = []
        for entry in payload:
            path = extract_path_from_entry(entry, processed_images_root)
            if path:
                paths.append(path)
        return paths

    if isinstance(payload, dict):
        for key in ("images", "data", "items", "entries"):
            value = payload.get(key)
            if isinstance(value, list):
                return collect_paths_from_payload(value, processed_images_root)

        if any(key in payload for key in ("image_path", "path_to_image", "path", "file_path", "image", "filename")):
            path = extract_path_from_entry(payload, processed_images_root)
            return [path] if path else []

        if all(key in payload for key in ("image_id", "patient_id", "study_id")):
            path = build_absolute_image_path(
                normalize_source_path(payload.get("patient_id")),
                normalize_source_path(payload.get("study_id")),
                normalize_source_path(payload.get("image_id")),
                processed_images_root,
            )
            return [path] if path else []

        path = extract_path_from_entry(payload, processed_images_root)
        if path:
            return [path]

        return [
            path
            for path in (extract_path_from_entry(value, processed_images_root) for value in payload.values())
            if path
        ]

    if isinstance(payload, str):
        path = normalize_image_path(payload, processed_images_root)
        return [path] if path else []

    return []


def extract_path_from_entry(entry: Any, processed_images_root: Path) -> str:
    if isinstance(entry, str):
        return normalize_image_path(entry, processed_images_root)

    if isinstance(entry, dict):
        if all(key in entry for key in ("image_id", "patient_id", "study_id")):
            built = build_absolute_image_path(
                normalize_source_path(entry.get("patient_id")),
                normalize_source_path(entry.get("study_id")),
                normalize_source_path(entry.get("image_id")),
                processed_images_root,
            )
            if built:
                return built

        for key in ("image_path", "path_to_image", "path", "file_path", "image", "filename"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return normalize_image_path(value, processed_images_root)

    return ""


def normalize_image_path(value: Any | None, processed_images_root: Path) -> str:
    raw = normalize_source_path(value)
    if not raw:
        return ""

    image_name = Path(raw.replace("\\", "/")).name
    image_stem = Path(image_name).stem
    suffix = Path(image_name).suffix or ".png"
    patient_id, study_id, _ = extract_image_triplet(raw)
    if not patient_id or not study_id or not image_stem:
        return raw

    return str((processed_images_root / patient_id / study_id / f"{image_stem}{suffix}").resolve())


def build_absolute_image_path(
    patient_id: str,
    study_id: str,
    image_id: str,
    processed_images_root: Path,
) -> str:
    patient = normalize_source_path(patient_id)
    study = normalize_source_path(study_id)
    image = normalize_source_path(image_id)
    if not patient or not study or not image:
        return ""

    return str((processed_images_root / patient / study / f"{image}.png").resolve())


def normalize_source_path(value: Any | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_label_value(value: Any | None) -> int:
    if value is None:
        return -100

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        numeric_value = int(float(value))
        if numeric_value == -1:
            return -100
        return numeric_value if numeric_value in {0, 1} else -100

    cleaned = str(value).strip()
    if not cleaned:
        return -100

    try:
        numeric_value = int(float(cleaned))
    except ValueError:
        return -100

    if numeric_value == -1:
        return -100
    return numeric_value if numeric_value in {0, 1} else -100


def normalize_report_text(raw_text: Any | None) -> str:
    if raw_text is None:
        return ""

    lines = [line.rstrip() for line in str(raw_text).splitlines()]
    non_empty_lines = [line for line in lines if line.strip()]
    return "\n".join(non_empty_lines).strip()


def compose_report(findings: str, impression: str) -> str:
    return f"FINDINGS: {findings}; IMPRESSION: {impression};"


def read_json_or_jsonl(path: Path) -> Any:
    raw_text = path.read_text(encoding="utf-8")
    stripped = raw_text.lstrip()
    if not stripped:
        return []

    if stripped[0] in "[{":
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

    rows: list[Any] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows