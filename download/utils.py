from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_dir / "preprocess.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def preprocess_image(image_path: Path, output_size: int) -> tuple[Image.Image, int, int]:
    """
    Resize image to short_edge=output_size, preserving aspect ratio.
    Returns: (resized_image, original_width, original_height)
    """
    with Image.open(image_path) as img:
        gray = img.convert("L")
        # Get original size before resize
        orig_w, orig_h = gray.size
        
        resized, _, _ = short_edge_resize(gray, output_size)

        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        result = Image.fromarray(arr_uint8, mode="L")
        return result, orig_w, orig_h


def short_edge_resize(image: Image.Image, short_edge_size: int) -> tuple[Image.Image, int, int]:
    """
    Resize image to have short edge = short_edge_size, long edge scales accordingly.
    Returns: (resized_image, width, height)
    """
    src_w, src_h = image.size
    
    # Calculate scale based on short edge
    if src_w < src_h:
        # Width is the short edge
        scale = short_edge_size / max(src_w, 1)
        new_w = short_edge_size
        new_h = max(1, int(round(src_h * scale)))
    else:
        # Height is the short edge
        scale = short_edge_size / max(src_h, 1)
        new_h = short_edge_size
        new_w = max(1, int(round(src_w * scale)))
    
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    return resized, new_w, new_h


def load_string_set(path: Path) -> set[str]:
    if not path.exists():
        return set()

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return set()

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return set()

    if not isinstance(payload, list):
        return set()

    return {str(item) for item in payload}


def save_string_set(path: Path, values: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(values), indent=2), encoding="utf-8")


def load_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, list):
        return []

    return [row for row in payload if isinstance(row, dict)]


def save_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def build_metadata_entry(
    image_id: str,
    patient_id: str,
    study_id: str,
    output_path: Path,
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "patient_id": patient_id,
        "study_id": study_id,
        "image_path": str(output_path),
    }


def save_image_dimensions_jsonl(
    dimensions_path: Path,
    image_id: str,
    patient_id: str,
    study_id: str,
    width: int,
    height: int,
    append: bool = True,
) -> None:
    """
    Save image dimensions to JSONL file (one JSON per line).
    If append=True, appends to existing file; otherwise overwrites.
    """
    dimensions_path.parent.mkdir(parents=True, exist_ok=True)
    
    entry = {
        "image_id": image_id,
        "patient_id": patient_id,
        "study_id": study_id,
        "width": width,
        "height": height,
    }
    
    mode = "a" if (append and dimensions_path.exists()) else "w"
    with dimensions_path.open(mode, encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
