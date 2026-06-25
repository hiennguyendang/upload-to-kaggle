from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

try:
    from preprocess.config import PipelineConfig
except ModuleNotFoundError:
    from config import PipelineConfig

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
ALLOWED_VIEWS = {"AP", "PA"}


@dataclass(frozen=True)
class MimicImageRecord:
    image_id: str
    patient_id: str
    visit_id: str
    p_folder: str
    image_path: Path


@dataclass(frozen=True)
class MimicScanStats:
    pack_name: str
    csv_rows_matched: int
    frontal_rows_matched: int
    unique_frontal_studies: int
    kept_images: int
    missing_images: int
    discarded_frontal_rows: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "pack_name": self.pack_name,
            "csv_rows_matched": self.csv_rows_matched,
            "frontal_rows_matched": self.frontal_rows_matched,
            "unique_frontal_studies": self.unique_frontal_studies,
            "kept_images": self.kept_images,
            "missing_images": self.missing_images,
            "discarded_frontal_rows": self.discarded_frontal_rows,
        }


def scan_mimic_images(
    mimic_root: Path,
    config: PipelineConfig,
    metadata_csv: Path,
    p_folder_filter: list[str] | None = None,
) -> tuple[list[MimicImageRecord], MimicScanStats]:
    """
    Select one AP/PA image per study based on mimic-cxr-2.0.0-metadata.csv.
    Priority within each study: PA first, then AP, then lexicographic dicom_id.
    """
    study_best_row: dict[tuple[str, str], tuple[int, str]] = {}
    csv_rows_matched = 0
    frontal_rows_matched = 0
    pack_name = p_folder_filter[0] if p_folder_filter and len(p_folder_filter) == 1 else "all"

    with metadata_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            return []

        header_index = {name: idx for idx, name in enumerate(header)}

        subject_idx = header_index.get("subject_id")
        study_idx = header_index.get("study_id")
        dicom_idx = header_index.get("dicom_id")
        view_idx = header_index.get("ViewPosition")

        if None in {subject_idx, study_idx, dicom_idx, view_idx}:
            return []

        for row in reader:
            if max(subject_idx, study_idx, dicom_idx, view_idx) >= len(row):
                continue

            subject_id = normalize_numeric_id(row[subject_idx])
            study_id = normalize_numeric_id(row[study_idx])
            dicom_id = normalize_token(row[dicom_idx])
            view = normalize_token(row[view_idx]).upper()

            if not subject_id or not study_id or not dicom_id or view not in ALLOWED_VIEWS:
                continue

            p_folder = infer_p_folder(subject_id)
            if p_folder_filter and p_folder not in p_folder_filter:
                continue

            csv_rows_matched += 1

            rank = view_rank(view)
            key = (subject_id, study_id)
            current = study_best_row.get(key)
            candidate = (rank, dicom_id)
            if current is None or candidate < current:
                study_best_row[key] = candidate

            if view in ALLOWED_VIEWS:
                frontal_rows_matched += 1

    records: list[MimicImageRecord] = []
    for (subject_id, study_id), (_, dicom_id) in sorted(study_best_row.items()):
        p_folder = infer_p_folder(subject_id)
        patient_id = f"p{subject_id}"
        visit_id = f"s{study_id}"
        study_dir = mimic_root / p_folder / patient_id / visit_id

        image_path = resolve_dicom_image_path(study_dir, dicom_id)
        if image_path is None:
            continue

        image_id = make_unique_mimic_image_id(p_folder, patient_id, visit_id, image_path)
        records.append(
            MimicImageRecord(
                image_id=image_id,
                patient_id=patient_id,
                visit_id=visit_id,
                p_folder=p_folder,
                image_path=image_path,
            )
        )

    unique_frontal_studies = len(study_best_row)
    kept_images = len(records)
    missing_images = max(unique_frontal_studies - kept_images, 0)
    discarded_frontal_rows = max(frontal_rows_matched - unique_frontal_studies, 0)

    stats = MimicScanStats(
        pack_name=pack_name,
        csv_rows_matched=csv_rows_matched,
        frontal_rows_matched=frontal_rows_matched,
        unique_frontal_studies=unique_frontal_studies,
        kept_images=kept_images,
        missing_images=missing_images,
        discarded_frontal_rows=discarded_frontal_rows,
    )

    return records, stats


def normalize_numeric_id(value: str | None) -> str:
    text = normalize_token(value)
    if not text:
        return ""
    return text.zfill(8) if text.isdigit() else ""


def normalize_token(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def infer_p_folder(subject_id: str) -> str:
    return f"p{subject_id[:2]}"


def view_rank(view: str) -> int:
    if view == "PA":
        return 0
    if view == "AP":
        return 1
    return 99


def resolve_dicom_image_path(study_dir: Path, dicom_id: str) -> Path | None:
    if not study_dir.exists() or not study_dir.is_dir():
        return None

    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = study_dir / f"{dicom_id}{suffix}"
        if candidate.exists() and candidate.is_file():
            return candidate

    for candidate in study_dir.glob(f"{dicom_id}.*"):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            return candidate

    return None


def make_unique_mimic_image_id(
    p_folder: str,
    patient_id: str,
    visit_id: str,
    image_path: Path,
) -> str:
    # .stem (khong duoi): output file la f"{image_id}.jpg" -> dung 1 duoi,
    # va extract_dicom_id(image_id) tra dung dicom (khop *_SceneGraph.json).
    return f"MIMIC_{patient_id}_{visit_id}_{image_path.stem}"
