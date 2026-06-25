from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

try:
    from preprocess.config import PipelineConfig, default_config
    from preprocess.download.mimic_dataset import MimicImageRecord, MimicScanStats, scan_mimic_images
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
    from config import PipelineConfig, default_config
    from preprocess.download.mimic_dataset import MimicImageRecord, MimicScanStats, scan_mimic_images
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
    parser = argparse.ArgumentParser(description="MIMIC-CXR preprocessing pipeline")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA"),
        help="Source directory containing p10, p11, ... pX folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\MIMIC-CXR"),
        help="Output directory for processed images and metadata",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\MIMIC-CXR\mimic-cxr-2.0.0-metadata.csv"),
        help="Path to mimic-cxr-2.0.0-metadata.csv used to select AP/PA images",
    )
    parser.add_argument(
        "--p-folders",
        nargs="+",
        default=None,
        help="Specific p-folders to process, e.g. p10 p11 p12 (process all if not specified)",
    )
    parser.add_argument(
        "--debug-max-images",
        type=int,
        default=None,
        help="Process only first N images for debugging",
    )
    parser.add_argument(
        "--delete-p-folders",
        action="store_true",
        help="Delete p-folders after successful processing to save disk space",
    )
    parser.add_argument(
        "--pack-name",
        type=str,
        default=None,
        help="Optional pack name (for example p10) used for resume tracking",
    )
    return parser.parse_args()


def build_mimic_config(args: argparse.Namespace) -> PipelineConfig:
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    output_images_dir = output_dir / "images"
    metadata_path = output_dir / "metadata.json"
    checkpoints_dir = output_dir / "checkpoints"
    processed_files_checkpoint = checkpoints_dir / "processed_files.json"
    processed_packs_checkpoint = checkpoints_dir / "processed_packs.json"
    logs_dir = output_dir / "logs"

    return PipelineConfig(
        work_root=output_dir,
        tmp_raw_dir=source_dir,
        processed_dir=output_dir,
        output_images_dir=output_images_dir,
        metadata_path=metadata_path,
        checkpoints_dir=checkpoints_dir,
        processed_files_checkpoint=processed_files_checkpoint,
        processed_packs_checkpoint=processed_packs_checkpoint,
        logs_dir=logs_dir,
        rclone_remote_root="",
        pack_names=(),
        debug_max_images=args.debug_max_images,
    )


def load_pack_summary(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def save_pack_summary(path: Path, summary: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def process_mimic_images(
    records: list[MimicImageRecord],
    config: PipelineConfig,
    processed_files: set[str],
    metadata_index: dict[str, dict],
    logger,
) -> tuple[int, int, int]:
    dimensions_path = config.processed_dir / "dimensions.jsonl"
    new_count = 0
    skipped_existing = 0
    failed_count = 0

    for record in tqdm(records, desc="Processing MIMIC images"):
        if record.image_id in processed_files:
            skipped_existing += 1
            continue

        try:
            processed, width, height = preprocess_image(record.image_path, config.output_size)
            out_dir = config.output_images_dir / record.p_folder / record.patient_id
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"{record.image_id}.jpg"
            processed.save(output_path, format="JPEG", quality=100, subsampling=0)

            metadata_index[record.image_id] = build_metadata_entry(
                image_id=record.image_id,
                patient_id=record.patient_id,
                study_id=record.visit_id,
                output_path=output_path,
            )

            save_image_dimensions_jsonl(
                dimensions_path=dimensions_path,
                image_id=record.image_id,
                patient_id=record.patient_id,
                study_id=record.visit_id,
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


def build_pack_summary_entry(
    *,
    stats: MimicScanStats,
    new_count: int,
    skipped_existing: int,
    failed_count: int,
    scan_elapsed: float,
    process_elapsed: float,
) -> dict:
    kept_total = new_count + skipped_existing
    dropped_total = max(stats.csv_rows_matched - kept_total, 0)
    return {
        "pack_name": stats.pack_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "csv_rows_matched": stats.csv_rows_matched,
        "frontal_rows_matched": stats.frontal_rows_matched,
        "unique_frontal_studies": stats.unique_frontal_studies,
        "kept_images": kept_total,
        "new_images": new_count,
        "skipped_existing": skipped_existing,
        "failed_images": failed_count,
        "missing_images": stats.missing_images,
        "discarded_frontal_rows": stats.discarded_frontal_rows,
        "dropped_rows_total": dropped_total,
        "scan_seconds": round(scan_elapsed, 3),
        "process_seconds": round(process_elapsed, 3),
    }


def mark_pack_completed(config: PipelineConfig, pack_name: str | None, completed_packs: set[str]) -> None:
    if not pack_name:
        return
    completed_packs.add(pack_name)
    save_string_set(config.processed_packs_checkpoint, completed_packs)


def main() -> None:
    args = parse_args()
    config = build_mimic_config(args)

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    metadata_csv = args.metadata_csv.resolve()

    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        return
    if not metadata_csv.exists():
        print(f"Metadata CSV not found: {metadata_csv}")
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
    logger.info("Starting MIMIC-CXR preprocessing")
    logger.info("Source: %s", source_dir)
    logger.info("Output: %s", output_dir)
    logger.info("Metadata CSV: %s", metadata_csv)

    processed_files = load_string_set(config.processed_files_checkpoint)
    completed_packs = load_string_set(config.processed_packs_checkpoint)
    summary_path = config.logs_dir / "mimic_pack_summary.json"
    pack_summary = load_pack_summary(summary_path)
    metadata_rows = load_metadata(config.metadata_path)
    metadata_index = {
        str(row["image_id"]): row
        for row in metadata_rows
        if isinstance(row, dict) and "image_id" in row
    }

    for image_id in metadata_index:
        processed_files.add(image_id)

    if args.pack_name and args.pack_name in completed_packs:
        logger.info("Pack already completed, skipping: %s", args.pack_name)
        return

    start_time = time.perf_counter()
    logger.info("Scanning MIMIC via metadata.csv (AP/PA only, one image per study)...")
    if args.p_folders:
        logger.info("Processing only p-folders: %s", ", ".join(args.p_folders))
        missing_p_folders = [name for name in args.p_folders if not (source_dir / name).exists()]
        if missing_p_folders:
            logger.warning(
                "Missing local p-folders: %s",
                ", ".join(missing_p_folders),
            )
            logger.warning(
                "If this is a rerun, previous run may have deleted source folders due to --delete-p-folders."
            )
    records, scan_stats = scan_mimic_images(
        source_dir,
        config,
        metadata_csv=metadata_csv,
        p_folder_filter=args.p_folders,
    )
    scan_elapsed = time.perf_counter() - start_time
    logger.info("Scan completed in %.1f sec, found %d candidate images", scan_elapsed, len(records))
    logger.info(
        "Pack stats | pack=%s | csv_matched=%d | frontal_rows=%d | unique_studies=%d | missing_images=%d",
        scan_stats.pack_name,
        scan_stats.csv_rows_matched,
        scan_stats.frontal_rows_matched,
        scan_stats.unique_frontal_studies,
        scan_stats.missing_images,
    )

    if args.pack_name and len(records) == 0:
        logger.warning("No candidate images found for pack %s", args.pack_name)

    if config.debug_max_images is not None:
        records = records[: config.debug_max_images]
        logger.info("Debug mode: processing only first %d images", len(records))

    process_start = time.perf_counter()
    new_count, skipped_existing, failed_count = process_mimic_images(
        records=records,
        config=config,
        processed_files=processed_files,
        metadata_index=metadata_index,
        logger=logger,
    )
    process_elapsed = time.perf_counter() - process_start

    save_string_set(config.processed_files_checkpoint, processed_files)
    save_metadata(config.metadata_path, list(metadata_index.values()))

    total_candidates = len(records)
    completed = new_count + skipped_existing
    pack_completed = total_candidates > 0 and failed_count == 0 and completed >= total_candidates

    if args.pack_name:
        pack_summary[args.pack_name] = build_pack_summary_entry(
            stats=scan_stats,
            new_count=new_count,
            skipped_existing=skipped_existing,
            failed_count=failed_count,
            scan_elapsed=scan_elapsed,
            process_elapsed=process_elapsed,
        )
        save_pack_summary(summary_path, pack_summary)

    if pack_completed:
        mark_pack_completed(config, args.pack_name, completed_packs)
        logger.info("Pack completed and checkpointed: %s", args.pack_name or "<none>")
    elif args.pack_name:
        logger.warning(
            "Pack not checkpointed because processing is incomplete (pack=%s, completed=%d/%d, failed=%d)",
            args.pack_name,
            completed,
            total_candidates,
            failed_count,
        )

    if args.delete_p_folders and args.p_folders:
        can_delete = True
        if config.debug_max_images is not None:
            logger.warning("Skip deleting source p-folders because debug mode is enabled")
            can_delete = False
        elif total_candidates == 0:
            logger.warning("Skip deleting source p-folders because scan found 0 images")
            can_delete = False
        elif failed_count > 0 or completed < total_candidates:
            logger.warning(
                "Skip deleting source p-folders because processing is incomplete (completed=%d/%d, failed=%d)",
                completed,
                total_candidates,
                failed_count,
            )
            can_delete = False

        if can_delete:
            for p_folder_name in args.p_folders:
                p_folder_path = source_dir / p_folder_name
                if p_folder_path.exists():
                    logger.info("Deleting p-folder: %s", p_folder_path)
                    shutil.rmtree(p_folder_path, ignore_errors=True)

    logger.info("MIMIC-CXR preprocessing finished")
    logger.info("Scan time: %.1f sec", scan_elapsed)
    logger.info("Process time: %.1f sec", process_elapsed)
    logger.info("Newly processed images: %d", new_count)
    logger.info("Skipped by checkpoint: %d", skipped_existing)
    logger.info("Failed images: %d", failed_count)
    logger.info("Total unique images: %d", len(processed_files))


if __name__ == "__main__":
    main()
