#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_REPO_ID = "zwhe99/DeepMath-103K"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_DIR = SCRIPT_DIR.parent.parent / "data" / "DeepMath-103K"
SHARDS = [f"data/train-{index:05d}-of-00010.parquet" for index in range(10)]


def display_path(path: Path | str) -> str:
    resolved = Path(path).expanduser().resolve()
    return os.path.relpath(resolved, SCRIPT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download DeepMath-103K parquet shards from Hugging Face.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_DIR))
    parser.add_argument(
        "--shards",
        default="0",
        help="Comma-separated shard indexes to download, or 'all'. Default downloads shard 0, enough for train100+val100.",
    )
    return parser.parse_args()


def selected_shards(value: str) -> list[str]:
    cleaned = value.strip().lower()
    if cleaned == "all":
        return SHARDS
    indexes = [int(part.strip()) for part in cleaned.split(",") if part.strip()]
    bad = [index for index in indexes if index < 0 or index >= len(SHARDS)]
    if bad:
        raise ValueError(f"Shard indexes must be in [0, 9], got {bad}")
    return [SHARDS[index] for index in indexes]


def main() -> None:
    args = parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("huggingface_hub is required to download DeepMath-103K") from exc

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required to inspect downloaded parquet shards") from exc

    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for filename in selected_shards(args.shards):
        path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
        )
        downloaded.append(path)
        rows = pq.ParquetFile(path).metadata.num_rows
        print(f"downloaded {display_path(path)} rows={rows}", flush=True)

    first = downloaded[0] if downloaded else None
    if first:
        print(f"default_train_dataset={display_path(first)}", flush=True)


if __name__ == "__main__":
    main()
