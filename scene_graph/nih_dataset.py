from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

NIH_NAME_PATTERN = re.compile(r"^(?P<patient_id>\d{8})_(?P<study_id>\d+)$")


@dataclass(frozen=True)
class NihImageRecord:
    image_id: str
    patient_id: str
    study_id: str
    split_name: str
    image_path: Path


def scan_nih_split_images(split_dir: Path) -> list[NihImageRecord]:
    records: list[NihImageRecord] = []

    images_dir = split_dir / "images"
    if not images_dir.exists() or not images_dir.is_dir():
        return records

    image_paths = sorted(
        [
            image_path
            for image_path in images_dir.iterdir()
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]
    )

    for image_path in image_paths:
        patient_id, study_id = extract_patient_and_study_from_name(image_path)
        image_id = make_unique_nih_image_id(
            split_name=split_dir.name,
            patient_id=patient_id,
            study_id=study_id,
            image_path=image_path,
        )

        records.append(
            NihImageRecord(
                image_id=image_id,
                patient_id=patient_id,
                study_id=study_id,
                split_name=split_dir.name,
                image_path=image_path,
            )
        )

    return records


def list_nih_split_dirs(source_dir: Path) -> list[Path]:
    return sorted(
        [
            split_dir
            for split_dir in source_dir.iterdir()
            if split_dir.is_dir() and re.fullmatch(r"(?:img|images)s?_\d{3}", split_dir.name)
        ]
    )


def extract_patient_and_study_from_name(image_path: Path) -> tuple[str, str]:
    match = NIH_NAME_PATTERN.match(image_path.stem)
    if match:
        return match.group("patient_id"), match.group("study_id")

    # Fallback for unexpected naming while keeping pipeline resilient.
    tokens = image_path.stem.split("_")
    if len(tokens) >= 2:
        return tokens[0], tokens[1]

    return "unknown_patient", "unknown_study"


def make_unique_nih_image_id(
    split_name: str,
    patient_id: str,
    study_id: str,
    image_path: Path,
) -> str:
    return f"NIH_{patient_id}_{study_id}_{image_path.name}"
