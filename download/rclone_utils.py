from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

try:
    from preprocess.config import PipelineConfig
except ModuleNotFoundError:
    from config import PipelineConfig


def run_rclone_copy(pack_name: str, config: PipelineConfig, logger) -> tuple[Path, float, float | None]:
    local_pack_dir = config.tmp_raw_dir / pack_name
    remote_pack = f"{config.rclone_remote_root}/{pack_name}"

    local_pack_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rclone",
        "copy",
        remote_pack,
        str(local_pack_dir),
        "-P",
        "--fast-list",
    ]

    start = time.perf_counter()

    for attempt in range(1, config.rclone_retry_count + 1):
        logger.info(
            "Downloading %s with rclone (attempt %d/%d)",
            pack_name,
            attempt,
            config.rclone_retry_count,
        )

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            elapsed = time.perf_counter() - start
            bytes_downloaded = compute_directory_size_bytes(local_pack_dir)
            speed_mb_s = None
            if elapsed > 0:
                speed_mb_s = (bytes_downloaded / (1024 * 1024)) / elapsed
            logger.info(
                "Downloaded %s in %.1f sec (%.2f MB/s)",
                pack_name,
                elapsed,
                speed_mb_s or 0.0,
            )
            return local_pack_dir, elapsed, speed_mb_s

        stderr_text = (proc.stderr or "").strip()
        stdout_text = (proc.stdout or "").strip()
        logger.error("rclone failed for %s", pack_name)
        if stdout_text:
            logger.error("rclone stdout: %s", stdout_text[-1000:])
        if stderr_text:
            logger.error("rclone stderr: %s", stderr_text[-1000:])

        if attempt < config.rclone_retry_count:
            wait_sec = config.rclone_retry_backoff_sec * attempt
            logger.warning("Retrying %s after %.1f sec", pack_name, wait_sec)
            time.sleep(wait_sec)

    raise RuntimeError(f"rclone copy failed for {pack_name} after {config.rclone_retry_count} attempts")


def remove_local_pack(pack_dir: Path, logger) -> None:
    if not pack_dir.exists():
        return
    logger.info("Removing local pack folder: %s", pack_dir)
    shutil.rmtree(pack_dir, ignore_errors=True)


def compute_directory_size_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total
