from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

try:
    from preprocess.config import PipelineConfig
except ModuleNotFoundError:
    from config import PipelineConfig

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    patient_id: str
    study_id: str
    pack_name: str
    image_path: Path


def scan_pack_images(pack_dir: Path, config: PipelineConfig) -> list[ImageRecord]:
    records: list[ImageRecord] = []

    for image_path in pack_dir.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue

        if infer_view(image_path, config.frontal_tokens) != "frontal":
            continue

        patient_id, study_id = extract_patient_and_study(image_path)
        image_id = make_unique_image_id(pack_dir.name, patient_id, study_id, image_path)

        records.append(
            ImageRecord(
                image_id=image_id,
                patient_id=patient_id,
                study_id=study_id,
                pack_name=pack_dir.name,
                image_path=image_path,
            )
        )

    return records


def infer_view(image_path: Path, frontal_tokens: tuple[str, ...]) -> str:
    stem = image_path.stem.lower()
    tokens = set(token for token in re.split(r"[^a-z0-9]+", stem) if token)

    if "lateral" in tokens or "lat" in tokens:
        return "lateral"

    for token in frontal_tokens:
        if token in tokens:
            return "frontal"

    return "unknown"


def extract_patient_and_study(image_path: Path) -> tuple[str, str]:
    patient_id = "unknown_patient"
    study_id = "unknown_study"

    patient_pattern = re.compile(r"^(patient\w+|p\d+)$", re.IGNORECASE)
    study_pattern = re.compile(r"^(study\w+|s\d+)$", re.IGNORECASE)

    for part in image_path.parts:
        if patient_pattern.match(part):
            patient_id = part
        if study_pattern.match(part):
            study_id = part

    return patient_id, study_id


def make_unique_image_id(
    pack_name: str,
    patient_id: str,
    study_id: str,
    image_path: Path,
) -> str:
    # .stem (khong duoi): output file la f"{image_id}.png" -> dung 1 duoi,
    # va dicom id (stem) khop voi label/report index khi build metadata.
    return f"{pack_name}_{patient_id}_{study_id}_{image_path.stem}"
