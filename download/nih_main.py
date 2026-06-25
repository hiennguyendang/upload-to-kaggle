from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from tqdm import tqdm

try:
    from preprocess.config import PipelineConfig
    from preprocess.download.nih_dataset import NihImageRecord, list_nih_split_dirs, scan_nih_split_images
    from preprocess.download.utils import (
        build_metadata_entry,
        ensure_directories,
        load_metadata,
        load_string_set,
        preprocess_image,
        save_image_dimensions_jsonl,
        save_metadata,
        save_string_set,
        setup_logger,
    )
except ModuleNotFoundError:
    from config import PipelineConfig
    from preprocess.download.nih_dataset import NihImageRecord, list_nih_split_dirs, scan_nih_split_images
    from preprocess.download.utils import (
        build_metadata_entry,
        ensure_directories,
        load_metadata,
        load_string_set,
        preprocess_image,
        save_image_dimensions_jsonl,
        save_metadata,
        save_string_set,
        setup_logger,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIH Chest X-ray preprocessing pipeline")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\NIH_RAW"),
        help="Source directory containing split folders imgs_001 ... imgs_012",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\NIH"),
        help="Output directory for resized images and metadata",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Specific split folders to process, e.g. imgs_001 imgs_002",
    )
    parser.add_argument(
        "--debug-max-images",
        type=int,
        default=None,
        help="Process only first N images for debugging",
    )
    parser.add_argument(
        "--keep-splits",
        action="store_true",
        help="Keep source split folders after processing",
    )
    return parser.parse_args()


def build_nih_config(args: argparse.Namespace) -> PipelineConfig:
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    output_images_dir = output_dir / "images"
    metadata_path = output_dir / "metadata.json"
    checkpoints_dir = output_dir / "checkpoints"
    processed_files_checkpoint = checkpoints_dir / "processed_files.json"
    processed_splits_checkpoint = checkpoints_dir / "processed_splits.json"
    logs_dir = output_dir / "logs"

    return PipelineConfig(
        work_root=output_dir,
        tmp_raw_dir=source_dir,
        processed_dir=output_dir,
        output_images_dir=output_images_dir,
        metadata_path=metadata_path,
        checkpoints_dir=checkpoints_dir,
        processed_files_checkpoint=processed_files_checkpoint,
        processed_packs_checkpoint=processed_splits_checkpoint,
        logs_dir=logs_dir,
        rclone_remote_root="",
        pack_names=(),
        debug_max_images=args.debug_max_images,
        delete_pack_after_process=not args.keep_splits,
    )


def process_nih_records(
    records: list[NihImageRecord],
    config: PipelineConfig,
    processed_files: set[str],
    metadata_index: dict[str, dict],
    logger,
) -> tuple[int, int, int]:
    dimensions_path = config.processed_dir / "dimensions.jsonl"
    new_count = 0
    skipped_existing = 0
    failed_count = 0

    for record in tqdm(records, desc=f"Processing {records[0].split_name}" if records else "Processing"):
        if record.image_id in processed_files:
            skipped_existing += 1
            continue

        try:
            processed, width, height = preprocess_image(record.image_path, config.output_size)

            # Keep all NIH outputs in one folder, not split-based subfolders.
            config.output_images_dir.mkdir(parents=True, exist_ok=True)
            output_path = config.output_images_dir / f"{record.image_id}.png"
            processed.save(output_path)

            metadata_index[record.image_id] = build_metadata_entry(
                image_id=record.image_id,
                patient_id=record.patient_id,
                study_id=record.study_id,
                output_path=output_path,
            )

            save_image_dimensions_jsonl(
                dimensions_path=dimensions_path,
                image_id=record.image_id,
                patient_id=record.patient_id,
                study_id=record.study_id,
                width=width,
                height=height,
            )

            processed_files.add(record.image_id)
            new_count += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("Skip corrupt image %s: %s", record.image_path, exc)
            failed_count += 1

        if new_count > 0 and new_count % 200 == 0:
            save_string_set(config.processed_files_checkpoint, processed_files)
            save_metadata(config.metadata_path, list(metadata_index.values()))

    return new_count, skipped_existing, failed_count


def delete_split_dir(split_dir: Path, logger) -> None:
    if split_dir.exists():
        logger.info("Deleting completed split: %s", split_dir)
        shutil.rmtree(split_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    config = build_nih_config(args)

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        return

    ensure_directories(
        [
            output_dir,
            config.output_images_dir,
            config.checkpoints_dir,
            config.logs_dir,
        ]
    )

    logger = setup_logger(config.logs_dir)
    logger.info("Starting NIH preprocessing")
    logger.info("Source: %s", source_dir)
    logger.info("Output: %s", output_dir)

    processed_files = load_string_set(config.processed_files_checkpoint)
    processed_splits = load_string_set(config.processed_packs_checkpoint)

    metadata_rows = load_metadata(config.metadata_path)
    metadata_index = {
        str(row["image_id"]): row
        for row in metadata_rows
        if isinstance(row, dict) and "image_id" in row
    }

    for image_id in metadata_index:
        processed_files.add(image_id)

    all_splits = list_nih_split_dirs(source_dir)
    if args.splits:
        requested = set(args.splits)
        split_dirs = [split_dir for split_dir in all_splits if split_dir.name in requested]
    else:
        split_dirs = all_splits

    logger.info("Detected split folders to process: %d", len(split_dirs))
    if args.splits:
        logger.info("Requested splits: %s", ", ".join(args.splits))

    total_new = 0
    total_skipped = 0
    total_failed = 0

    global_start = time.perf_counter()

    for split_dir in split_dirs:
        split_name = split_dir.name
        if split_name in processed_splits:
            logger.info("Skip %s because it is already completed", split_name)
            continue

        split_start = time.perf_counter()
        logger.info("Scanning split: %s", split_name)
        records = scan_nih_split_images(split_dir)

        if config.debug_max_images is not None:
            records = records[: config.debug_max_images]
            logger.info("Debug mode for %s: processing only first %d images", split_name, len(records))

        logger.info("%s scanned images: %d", split_name, len(records))

        new_count, skipped_existing, failed_count = process_nih_records(
            records=records,
            config=config,
            processed_files=processed_files,
            metadata_index=metadata_index,
            logger=logger,
        )

        total_new += new_count
        total_skipped += skipped_existing
        total_failed += failed_count

        processed_splits.add(split_name)
        save_string_set(config.processed_files_checkpoint, processed_files)
        save_string_set(config.processed_packs_checkpoint, processed_splits)
        save_metadata(config.metadata_path, list(metadata_index.values()))

        if config.delete_pack_after_process:
            delete_split_dir(split_dir, logger)

        split_elapsed = time.perf_counter() - split_start
        logger.info(
            "%s done | new=%d skipped=%d failed=%d | elapsed=%.1f sec",
            split_name,
            new_count,
            skipped_existing,
            failed_count,
            split_elapsed,
        )

    total_elapsed = time.perf_counter() - global_start

    logger.info("NIH preprocessing finished")
    logger.info("Total elapsed: %.1f sec", total_elapsed)
    logger.info("Total newly processed images: %d", total_new)
    logger.info("Total skipped by checkpoint: %d", total_skipped)
    logger.info("Total failed images: %d", total_failed)
    logger.info("Total unique processed images: %d", len(processed_files))
    logger.info("Completed splits: %d", len(processed_splits))


if __name__ == "__main__":
    main()
