from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.metadata.mimic_metadata import build_mimic_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standardized MIMIC metadata JSONL")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\MIMIC-CXR"),
        help="Root directory of the MIMIC-CXR dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for JSONL and helper files (defaults to <dataset-root>/metadata)",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Directory containing processed MIMIC images",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=None,
        help="Directory containing MIMIC report text files",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Path to mimic-cxr-2.0.0-chexpert.csv",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Path to mimic-cxr-2.0.0-metadata.csv",
    )
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("mimic_metadata")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def main() -> int:
    args = parse_args()
    logger = setup_logger()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else None

    build_mimic_metadata(
        dataset_root=dataset_root,
        output_dir=output_dir,
        images_root=args.images_root,
        report_root=args.report_root,
        labels_csv=args.labels_csv,
        metadata_csv=args.metadata_csv,
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
