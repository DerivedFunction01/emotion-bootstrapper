#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from emotion_cache import load_json, load_parquet_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect saved bootstrap chunk parquet files for resume debugging."
    )
    parser.add_argument("--work-dir", default="./bootstrap_work")
    parser.add_argument("--chunk-dir", default=None, help="Optional explicit chunk directory")
    parser.add_argument("--limit", type=int, default=5, help="Number of sample rows to print per chunk")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    cache_dir = work_dir / "cache"
    chunk_dir = Path(args.chunk_dir) if args.chunk_dir else work_dir / "chunks"

    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    chunk_files = sorted(chunk_dir.glob("chunk_*.parquet"))
    print(f"Chunk directory: {chunk_dir}")
    print(f"Chunk files found: {len(chunk_files)}")

    if (cache_dir / "cache_meta.json").exists():
        meta = load_json(cache_dir / "cache_meta.json")
        print("\nCache metadata:")
        print(f"  dataset_path: {meta.get('dataset_path')}")
        print(f"  dataset_config: {meta.get('dataset_config')}")
        print(f"  text_column: {meta.get('text_column')}")
        print(f"  num_rows: {meta.get('num_rows')}")
        print(f"  num_hypotheses: {meta.get('num_hypotheses')}")
    else:
        print("\nCache metadata: not found")

    if (cache_dir / "tokenized.parquet").exists():
        tokenized = load_parquet_dataset(cache_dir / "tokenized.parquet")
        print("\nTokenized parquet:")
        print(f"  rows: {len(tokenized)}")
        print(f"  columns: {list(tokenized.column_names)}")
        print(f"  text_index range: {min(tokenized['text_index']) if len(tokenized) else 'n/a'}..{max(tokenized['text_index']) if len(tokenized) else 'n/a'}")
    else:
        print("\nTokenized parquet: not found")

    total_rows = 0
    for path in chunk_files:
        df = pd.read_parquet(path)
        n = len(df)
        total_rows += n
        print(f"\n{path.name}: rows={n}, columns={list(df.columns)}")
        print("Sample rows:")
        for _, row in df.head(args.limit).iterrows():
            print(row.to_dict())

    print(f"\nTotal rows across all chunks: {total_rows}")


if __name__ == "__main__":
    main()
