from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

try:
    from preprocess.config import PipelineConfig, default_config
    from preprocess.chexplus_dataset import scan_pack_images
    from preprocess.download.rclone_utils import remove_local_pack, run_rclone_copy
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
    from chexplus_dataset import scan_pack_images
    from preprocess.download.rclone_utils import remove_local_pack, run_rclone_copy
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
    parser = argparse.ArgumentParser(description="Rclone-based chest X-ray preprocessing")
    parser.add_argument("--work-root", type=Path, default=Path("C:/workspace"))
    parser.add_argument("--max-packs", type=int, default=None)
    parser.add_argument("--debug-max-images", type=int, default=None)
    parser.add_argument("--keep-pack", action="store_true")
    parser.add_argument(
        "--local-pack-root",
        type=Path,
        default=None,
        help="Process existing local PACK folders from this directory instead of rclone",
    )
    parser.add_argument(
        "--packs",
        nargs="+",
        default=None,
        help="Explicit pack names to process, e.g. PACK_1 PACK_3 PACK_4",
    )
    parser.add_argument(
        "--rclone-remote-root",
        type=str,
        default="dhint:CHEX-DATA/CheXplus",
        help="Remote root containing PACK folders",
    )
    parser.add_argument("--rclone-retries", type=int, default=3)
    parser.add_argument("--rclone-backoff-sec", type=float, default=3.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="parallel threads for image resize+save (1 = serial)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    cfg = default_config(
        work_root=args.work_root,
        max_packs=args.max_packs,
        debug_max_images=args.debug_max_images,
        delete_pack_after_process=not args.keep_pack,
        rclone_remote_root=args.rclone_remote_root,
    )

    return replace(
        cfg,
        rclone_retry_count=max(1, args.rclone_retries),
        rclone_retry_backoff_sec=max(0.0, args.rclone_backoff_sec),
    )


def _resize_and_save(record, config: PipelineConfig):
    """CPU/IO-heavy part, safe to run in a thread (writes a unique file per image)."""
    processed, width, height = preprocess_image(record.image_path, config.output_size)
    out_dir = config.output_images_dir / record.patient_id / record.study_id
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{record.image_id}.png"
    processed.save(output_path)
    return width, height, output_path


def process_one_pack(
    pack_name: str,
    local_pack_dir: Path,
    config: PipelineConfig,
    processed_files: set[str],
    metadata_index: dict[str, dict],
    logger,
    workers: int = 8,
) -> int:
    dimensions_path = config.processed_dir / "dimensions.jsonl"
    records = scan_pack_images(local_pack_dir, config)
    logger.info("%s scanned: %d frontal images", pack_name, len(records))

    if config.debug_max_images is not None:
        records = records[: config.debug_max_images]
        logger.info("Debug mode for %s: process only first %d images", pack_name, len(records))

    todo = [r for r in records if r.image_id not in processed_files]
    new_count = 0

    def _bookkeep(record, width, height, output_path) -> None:
        # runs on the main thread only -> shared state stays consistent
        nonlocal new_count
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
        if new_count % 200 == 0:
            save_string_set(config.processed_files_checkpoint, processed_files)
            save_metadata(config.metadata_path, list(metadata_index.values()))

    if workers <= 1:
        for record in tqdm(todo, desc=f"Processing {pack_name}"):
            try:
                width, height, output_path = _resize_and_save(record, config)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Skip corrupt image %s: %s", record.image_path, exc)
                continue
            _bookkeep(record, width, height, output_path)
        return new_count

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_resize_and_save, r, config): r for r in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {pack_name}"):
            record = futures[fut]
            try:
                width, height, output_path = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Skip corrupt image %s: %s", record.image_path, exc)
                continue
            _bookkeep(record, width, height, output_path)

    return new_count


def process_local_pack(
    pack_name: str,
    pack_dir: Path,
    config: PipelineConfig,
    processed_files: set[str],
    metadata_index: dict[str, dict],
    logger,
    workers: int = 8,
) -> int:
    if not pack_dir.exists():
        raise FileNotFoundError(f"Local pack folder not found: {pack_dir}")

    logger.info("Using existing local pack: %s", pack_dir)
    return process_one_pack(
        pack_name=pack_name,
        local_pack_dir=pack_dir,
        config=config,
        processed_files=processed_files,
        metadata_index=metadata_index,
        logger=logger,
        workers=workers,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    ensure_directories(
        [
            config.work_root,
            config.tmp_raw_dir,
            config.processed_dir,
            config.output_images_dir,
            config.checkpoints_dir,
            config.logs_dir,
        ]
    )

    logger = setup_logger(config.logs_dir)
    logger.info("Starting rclone pack pipeline")
    logger.info("Work root: %s", config.work_root)
    logger.info("Remote root: %s", config.rclone_remote_root)
    if args.local_pack_root is not None:
        logger.info("Local pack root: %s", args.local_pack_root)
    if args.packs:
        logger.info("Explicit packs: %s", ", ".join(args.packs))

    processed_files = load_string_set(config.processed_files_checkpoint)
    processed_packs = load_string_set(config.processed_packs_checkpoint)

    metadata_rows = load_metadata(config.metadata_path)
    metadata_index = {
        str(row["image_id"]): row
        for row in metadata_rows
        if isinstance(row, dict) and "image_id" in row
    }

    for image_id in metadata_index:
        processed_files.add(image_id)

    packs = list(args.packs) if args.packs else list(config.pack_names)
    if config.max_packs is not None:
        packs = packs[: config.max_packs]

    total_new_images = 0

    for pack_name in packs:
        if pack_name in processed_packs:
            logger.info("Skip %s because it is already completed", pack_name)
            continue

        pack_total_start = time.perf_counter()
        local_pack_dir = config.tmp_raw_dir / pack_name

        try:
            if args.local_pack_root is not None:
                local_pack_dir = args.local_pack_root / pack_name
                download_sec = 0.0
                speed_mb_s = None
                new_images = process_local_pack(
                    pack_name=pack_name,
                    pack_dir=local_pack_dir,
                    config=config,
                    processed_files=processed_files,
                    metadata_index=metadata_index,
                    logger=logger,
                    workers=args.workers,
                )
            else:
                local_pack_dir, download_sec, speed_mb_s = run_rclone_copy(pack_name, config, logger)
                logger.info(
                    "%s download summary: %.1f sec, %.2f MB/s",
                    pack_name,
                    download_sec,
                    speed_mb_s or 0.0,
                )

                new_images = process_one_pack(
                    pack_name=pack_name,
                    local_pack_dir=local_pack_dir,
                    config=config,
                    processed_files=processed_files,
                    metadata_index=metadata_index,
                    logger=logger,
                    workers=args.workers,
                )

            total_new_images += new_images
            logger.info("%s processed images: %d", pack_name, new_images)

            processed_packs.add(pack_name)
            save_string_set(config.processed_files_checkpoint, processed_files)
            save_string_set(config.processed_packs_checkpoint, processed_packs)
            save_metadata(config.metadata_path, list(metadata_index.values()))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pack %s failed: %s", pack_name, exc)
        finally:
            if config.delete_pack_after_process or args.local_pack_root is not None:
                remove_local_pack(local_pack_dir, logger)

        pack_elapsed = time.perf_counter() - pack_total_start
        logger.info("%s total elapsed time: %.1f sec", pack_name, pack_elapsed)

    logger.info("Pipeline finished")
    logger.info("Total newly processed images this run: %d", total_new_images)
    logger.info("Total processed unique images: %d", len(processed_files))
    logger.info("Completed packs: %d", len(processed_packs))


if __name__ == "__main__":
    main()
