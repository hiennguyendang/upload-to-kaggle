from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.metadata.chexplus_metadata import build_chexplus_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standardized CheXplus metadata JSONL")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\CHEXPLUS"),
        help="Root directory of the CheXplus dataset",
    )
    parser.add_argument(
        "--images-source",
        type=Path,
        required=True,
        help="JSON/JSONL/TXT manifest containing image paths",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\CHEXPLUS\df_chexpert_plus_240401.csv"),
        help="Path to df_chexpert_plus_240401.csv",
    )
    parser.add_argument(
        "--labels-json",
        type=Path,
        required=True,
        help="Path to the CheXplus labels JSON/JSONL file",
    )
    parser.add_argument(
        "--processed-images-root",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\CHEXPLUS\processed\images"),
        help="Root folder where processed images are stored",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="chexplus_unified.jsonl",
        help="Output JSONL filename",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the JSONL file (defaults to <dataset-root>/metadata)",
    )
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("chexplus_metadata")
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
    output_dir = args.output_dir.resolve() if args.output_dir is not None else dataset_root / "metadata"

    build_chexplus_metadata(
        images_source=args.images_source,
        csv_path=args.csv_path,
        labels_json=args.labels_json,
        output_dir=output_dir,
        processed_images_root=args.processed_images_root,
        output_filename=args.output_filename,
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())