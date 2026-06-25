from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    work_root: Path
    tmp_raw_dir: Path
    processed_dir: Path
    output_images_dir: Path
    metadata_path: Path
    checkpoints_dir: Path
    processed_files_checkpoint: Path
    processed_packs_checkpoint: Path
    logs_dir: Path
    rclone_remote_root: str
    pack_names: tuple[str, ...]
    output_size: int = 512
    frontal_tokens: tuple[str, ...] = ("frontal", "pa", "ap")
    max_packs: int | None = None
    debug_max_images: int | None = None
    delete_pack_after_process: bool = True
    rclone_retry_count: int = 3
    rclone_retry_backoff_sec: float = 3.0


def default_config(
    work_root: Path | None = None,
    max_packs: int | None = None,
    debug_max_images: int | None = None,
    delete_pack_after_process: bool = True,
    rclone_remote_root: str = "gdrive:CHEX-DATA/CheXplus",
) -> PipelineConfig:
    base = Path("D:/workspace") if work_root is None else Path(work_root)

    tmp_raw_dir = base / "tmp_raw"
    processed_dir = base / "processed"
    output_images_dir = processed_dir / "images"
    metadata_path = processed_dir / "metadata.json"

    checkpoints_dir = base / "checkpoints"
    processed_files_checkpoint = checkpoints_dir / "processed_files.json"
    processed_packs_checkpoint = checkpoints_dir / "processed_packs.json"

    logs_dir = base / "logs"

    pack_names = tuple(f"PACK_{idx}" for idx in range(5))

    return PipelineConfig(
        work_root=base,
        tmp_raw_dir=tmp_raw_dir,
        processed_dir=processed_dir,
        output_images_dir=output_images_dir,
        metadata_path=metadata_path,
        checkpoints_dir=checkpoints_dir,
        processed_files_checkpoint=processed_files_checkpoint,
        processed_packs_checkpoint=processed_packs_checkpoint,
        logs_dir=logs_dir,
        rclone_remote_root=rclone_remote_root,
        pack_names=pack_names,
        max_packs=max_packs,
        debug_max_images=debug_max_images,
        delete_pack_after_process=delete_pack_after_process,
    )
