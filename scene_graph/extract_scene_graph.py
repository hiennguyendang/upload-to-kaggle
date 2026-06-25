"""
Build CheXplus scene-graph CSV from RadGraph section JSONs and df_chexpert_plus_240401.csv.

Join strategy:
- section_findings.json rows are aligned with the CSV rows where section_findings is non-empty.
- section_impression.json rows are aligned with the CSV rows where section_impression is non-empty.
- patient_id and study_id are derived from the corresponding CSV path_to_image field.

Output:
- C:/Users/dhint/CHEX-DATA/CHEXPLUS/metadata/chexplus_scene_graph.csv
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from preprocess.metadata.chexplus_metadata import extract_image_triplet


LOGGER = logging.getLogger("chexplus_scene_graph")
TEMPORAL_MAP: dict[str, list[str]] = {
    "worsened": [
        "increased",
        "increasing",
        "worsened",
        "worsening",
        "greater",
        "progressive",
        "enlarging",
        "prominent",
        "worse",
        "elevated",
    ],
    "improved": [
        "decreased",
        "decreasing",
        "improved",
        "improving",
        "resolved",
        "resolving",
        "clearing",
        "less",
        "smaller",
        "reduced",
    ],
    "stable": [
        "stable",
        "unchanged",
        "persistent",
        "persists",
        "similar",
        "no change",
        "baseline",
        "constant",
    ],
    "new": ["new", "newly", "developed", "developing", "emerging"],
}

STUDY_KEY_RE = re.compile(r"p(?P<patient>\d+)[_\-]?s(?P<study>\d+)", re.IGNORECASE)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("chexplus_scene_graph")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def normalize_presence(label: str) -> str | None:
    label_text = normalize_text(label).lower()
    if not label_text:
        return None

    suffix = label_text.split("::")[-1]
    if "present" in suffix:
        return "present"
    if "absent" in suffix:
        return "absent"
    if "uncertain" in suffix or "possible" in suffix or "suspect" in suffix:
        return "uncertain"
    return suffix or None


def normalize_temporal(token: str) -> str | None:
    token_text = normalize_text(token).lower()
    if not token_text:
        return None

    for temporal_status, keywords in TEMPORAL_MAP.items():
        if any(keyword in token_text for keyword in keywords):
            return temporal_status
    return None


def extract_study_id_from_key(raw_key: str) -> str:
    raw_text = normalize_text(raw_key)
    if not raw_text:
        return raw_text
    match = STUDY_KEY_RE.search(raw_text)
    if match:
        return f"p{match.group('patient')}_s{match.group('study')}"
    return raw_text


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}, found {type(payload).__name__}")
    return [item for item in payload if isinstance(item, dict)]


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return [row for row in reader]


def build_section_rows(
    json_rows: list[dict[str, Any]],
    csv_rows: list[dict[str, str]],
    section_name: str,
    section_field: str,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    matching_csv_rows = [row for row in csv_rows if normalize_text(row.get(section_field))]

    if len(json_rows) != len(matching_csv_rows):
        logger.warning(
            "%s count mismatch: JSON=%d, CSV rows with %s=%d. Using min length.",
            section_name,
            len(json_rows),
            section_field,
            len(matching_csv_rows),
        )

    pair_count = min(len(json_rows), len(matching_csv_rows))
    records: list[dict[str, Any]] = []

    for idx in range(pair_count):
        json_item = json_rows[idx]
        csv_item = matching_csv_rows[idx]
        payload = next(iter(json_item.values())) if len(json_item) == 1 else json_item
        if not isinstance(payload, dict):
            continue

        path_text = normalize_text(csv_item.get("path_to_image"))
        patient_id, study_id, view = extract_image_triplet(path_text)
        if not patient_id or not study_id:
            continue

        study_key = f"{patient_id}_{study_id}"
        records.extend(
            extract_observation_rows(
                payload=payload,
                patient_id=patient_id,
                study_id=study_id,
                study_key=study_key,
                source_section=section_name,
                source_view=view,
            )
        )

    return records


def extract_observation_rows(
    payload: dict[str, Any],
    patient_id: str,
    study_id: str,
    study_key: str,
    source_section: str,
    source_view: str,
) -> list[dict[str, Any]]:
    entities_raw = payload.get("entities") or {}
    if not isinstance(entities_raw, dict):
        return []

    entities = {str(entity_id): entity for entity_id, entity in entities_raw.items() if isinstance(entity, dict)}

    located_by_target: dict[str, list[str]] = {}
    modified_by_target: dict[str, list[str]] = {}
    for entity_id, entity in entities.items():
        for relation in entity.get("relations") or []:
            if not isinstance(relation, (list, tuple)) or len(relation) < 2:
                continue
            relation_name = normalize_text(relation[0]).lower()
            target_id = str(relation[1])
            if relation_name == "located_at":
                located_by_target.setdefault(target_id, []).append(entity_id)
            elif relation_name == "modify":
                modified_by_target.setdefault(target_id, []).append(entity_id)

    rows: list[dict[str, Any]] = []
    for entity_id, entity in entities.items():
        label = normalize_text(entity.get("label"))
        if not label.lower().startswith("observation"):
            continue

        observation = normalize_text(entity.get("tokens") or entity.get("text"))
        if not observation:
            continue

        presence = normalize_presence(label)

        anatomy = None
        for target_id in (relation[1] for relation in entity.get("relations") or [] if isinstance(relation, (list, tuple)) and len(relation) >= 2 and normalize_text(relation[0]).lower() == "located_at"):
            target_node = entities.get(str(target_id))
            if not target_node:
                continue
            target_label = normalize_text(target_node.get("label"))
            if target_label.lower().startswith("anatomy"):
                anatomy = normalize_text(target_node.get("tokens") or target_node.get("text"))
                if anatomy:
                    break

        temporal_status = None
        for modifier_id in modified_by_target.get(entity_id, []):
            modifier_node = entities.get(modifier_id)
            if not modifier_node:
                continue
            modifier_token = normalize_text(modifier_node.get("tokens") or modifier_node.get("text"))
            temporal_status = normalize_temporal(modifier_token)
            if temporal_status:
                break

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
                "bboxes": [],
            }
        )

    return rows


def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    dedupe_columns = ["patient_id", "study_id", "anatomy", "observation"]
    priority_map = {"impression": 0, "findings": 1}

    work_df = df.copy()
    work_df["priority"] = work_df["source_section"].map(priority_map).fillna(2).astype(int)
    work_df = work_df.sort_values(dedupe_columns + ["priority"], kind="mergesort")
    work_df = work_df.drop_duplicates(subset=dedupe_columns, keep="first")
    return work_df.drop(columns=["priority"])


def is_study_one(series: pd.Series) -> pd.Series:
    study_text = series.fillna("").astype(str).str.strip().str.lower()
    return study_text.str.fullmatch(r"(?:study|s)?0*1")


def main() -> None:
    logger = setup_logger()

    base_dir = Path(r"C:\Users\dhint\CHEX-DATA\CHEXPLUS")
    csv_path = base_dir / "df_chexpert_plus_240401.csv"
    findings_path = base_dir / "radgraph-XL-annotations" / "section_findings.json"
    impression_path = base_dir / "radgraph-XL-annotations" / "section_impression.json"
    output_dir = base_dir / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "chexplus_scene_graph.csv"

    csv_rows = load_csv_rows(csv_path)
    findings_json_rows = load_json_list(findings_path)
    impression_json_rows = load_json_list(impression_path)

    logger.info("Loaded CSV rows: %d", len(csv_rows))
    logger.info("Loaded findings JSON rows: %d", len(findings_json_rows))
    logger.info("Loaded impression JSON rows: %d", len(impression_json_rows))

    all_rows: list[dict[str, Any]] = []
    all_rows.extend(build_section_rows(findings_json_rows, csv_rows, "findings", "section_findings", logger))
    all_rows.extend(build_section_rows(impression_json_rows, csv_rows, "impression", "section_impression", logger))

    logger.info("Extracted rows before dedup: %d", len(all_rows))
    df = pd.DataFrame(all_rows)
    if df.empty:
        logger.warning("No rows extracted; nothing to write.")
        return

    for col in ("patient_id", "study_id", "study_key", "source_section", "source_view", "anatomy", "observation", "presence", "temporal_status", "bboxes"):
        if col not in df.columns:
            df[col] = None

    before_dedup = len(df)
    df = deduplicate_rows(df)
    after_dedup = len(df)

    # Rule: all study1 rows must have empty temporal_status.
    df.loc[is_study_one(df["study_id"]), "temporal_status"] = None

    # Stable, readable column order.
    output_columns = [
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
    df = df[output_columns]

    df.to_csv(output_csv, index=False)

    logger.info("Rows before dedup: %d", before_dedup)
    logger.info("Rows after dedup: %d", after_dedup)
    logger.info("Temporal status distribution:")
    print(df["temporal_status"].value_counts(dropna=False))
    logger.info("Wrote CSV to %s", output_csv)


if __name__ == "__main__":
    main()
