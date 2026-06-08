#!/usr/bin/env python3
"""
Fix Hindi 'joy' hypothesis scores in existing parquet chunks.
Matches rows by text, re-infers the joy hypothesis, and replaces the score.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

CORRECTED_HINDI_JOY = (
    "किसी को कम से कम बड़े आनंद और खुशी की एक मजबूत भावना महसूस हो रही है"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix Hindi joy hypothesis scores in emotion chunks."
    )
    parser.add_argument(
        "--chunks-dir", required=True, help="Directory containing chunk_*.parquet files"
    )
    parser.add_argument(
        "--output-parquet", required=True, help="Final output parquet path to patch"
    )
    parser.add_argument(
        "--server-url",
        required=True,
        help="Emotion inference server URL (e.g., http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write, just report what would change",
    )
    return parser.parse_args()


def _post_json(
    url: str, payload: dict[str, Any], timeout: float = 300.0
) -> dict[str, Any]:
    """POST JSON to server and get response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Server request failed for {url}: {exc.code} {detail}"
        ) from exc


def load_original_hindi_texts() -> set[str]:
    """Load the original mt-emotions dataset and extract Hindi texts."""
    print("Loading original DerivedFunction01/mt-emotions dataset...")
    try:
        dataset_dict = load_dataset("DerivedFunction01/mt-emotions")
        dataset = (
            dataset_dict["train"]
            if "train" in dataset_dict
            else next(iter(dataset_dict.values()))
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("You may need to be authenticated or have network access.")
        raise

    # Filter for Hindi rows
    if "translation_language" in dataset.column_names:
        hindi_rows = dataset.filter(lambda row: row["translation_language"] == "hindi")
    else:
        hindi_rows = dataset

    hindi_texts = set(hindi_rows["text"])
    print(f"Found {len(hindi_texts)} Hindi texts in original dataset")
    return hindi_texts


def find_hindi_rows_in_chunks(
    chunks_dir: Path, hindi_texts: set[str]
) -> dict[Path, list[int]]:
    """Find which chunks and rows contain Hindi texts."""
    chunk_paths = sorted(chunks_dir.glob("chunk_*.parquet"))
    hindi_matches: dict[Path, list[int]] = {}

    print(f"\nSearching {len(chunk_paths)} chunks for Hindi texts...")
    for chunk_path in tqdm(chunk_paths, desc="Scanning chunks"):
        df = pd.read_parquet(chunk_path)
        if "text" not in df.columns:
            continue

        matching_rows = []
        for idx, text in enumerate(df["text"]):
            if str(text) in hindi_texts:
                matching_rows.append(idx)

        if matching_rows:
            hindi_matches[chunk_path] = matching_rows

    total_hindi = sum(len(rows) for rows in hindi_matches.values())
    print(f"Found {total_hindi} Hindi rows across {len(hindi_matches)} chunks")
    return hindi_matches


def infer_joy_scores(
    server_url: str, texts: list[str], batch_size: int = 32
) -> dict[str, float]:
    """Infer joy scores for texts using the server."""
    # Load tokenizer to tokenize premise/hypothesis pairs
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

    # Create premise/hypothesis pairs for joy only
    pair_texts = texts
    pair_hypotheses = [CORRECTED_HINDI_JOY] * len(texts)

    # Tokenize all pairs
    tokenized = tokenizer(
        pair_texts,
        pair_hypotheses,
        padding=True,
        truncation=True,
        max_length=512,
        return_dict=True,
    )

    # Send to server in batches
    all_scores = []
    for start in tqdm(
        range(0, len(tokenized["input_ids"]), batch_size),
        desc="Inferring joy scores",
        total=(len(tokenized["input_ids"]) + batch_size - 1) // batch_size,
    ):
        end = min(start + batch_size, len(tokenized["input_ids"]))
        batch = {
            "examples": [
                {
                    "input_ids": tokenized["input_ids"][i],
                    "attention_mask": tokenized["attention_mask"][i],
                }
                for i in range(start, end)
            ]
        }
        response = _post_json(f"{server_url.rstrip('/')}/infer", batch)
        scores = response.get("entailment_scores", [])
        all_scores.extend(scores)

    # Map back to original texts
    text_to_score = {text: float(score) for text, score in zip(texts, all_scores)}
    return text_to_score


def update_emotion_vector(emotion_vector: dict, joy_score: float) -> dict:
    """Update emotion_vector with new joy score."""
    updated = dict(emotion_vector)
    updated["joy"] = joy_score
    return updated


def fix_chunks(
    chunks_dir: Path,
    hindi_matches: dict[Path, list[int]],
    server_url: str,
    batch_size: int = 32,
    dry_run: bool = False,
) -> None:
    """Fix joy scores in chunks."""
    total_fixed = 0

    for chunk_path, row_indices in tqdm(hindi_matches.items(), desc="Fixing chunks"):
        df = pd.read_parquet(chunk_path)

        # Extract texts and infer joy scores
        texts_to_infer = [df.iloc[idx]["text"] for idx in row_indices]
        text_to_joy_score = infer_joy_scores(server_url, texts_to_infer, batch_size)

        # Update emotion vectors
        for idx in row_indices:
            old_emotion_vector = df.loc[idx, "emotion_vector"]
            new_emotion_vector = update_emotion_vector(
                old_emotion_vector, text_to_joy_score[df.iloc[idx]["text"]]
            )
            df.loc[idx, "emotion_vector"] = new_emotion_vector

        if not dry_run:
            df.to_parquet(chunk_path, index=False)
            print(f"Updated {chunk_path}: {len(row_indices)} rows fixed")

        total_fixed += len(row_indices)

    print(f"\nTotal rows fixed: {total_fixed}")
    return total_fixed


def rebuild_final_parquet(
    chunks_dir: Path, output_parquet: Path, dry_run: bool = False
) -> None:
    """Rebuild final parquet from updated chunks."""
    if dry_run:
        print(f"\n[DRY RUN] Would rebuild {output_parquet} from chunks in {chunks_dir}")
        return

    chunk_paths = sorted(chunks_dir.glob("chunk_*.parquet"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunks found in {chunks_dir}")

    print(f"Rebuilding {output_parquet} from {len(chunk_paths)} chunks...")
    dfs = [pd.read_parquet(path) for path in tqdm(chunk_paths, desc="Loading chunks")]
    final_df = pd.concat(dfs, ignore_index=True)

    # Atomic write
    tmp_path = output_parquet.with_suffix(output_parquet.suffix + ".tmp")
    final_df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, output_parquet)
    print(f"Saved {output_parquet}")


def main() -> None:
    args = parse_args()
    chunks_dir = Path(args.chunks_dir)
    output_parquet = Path(args.output_parquet)

    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    # Step 1: Load original Hindi texts
    hindi_texts = load_original_hindi_texts()

    # Step 2: Find Hindi rows in chunks
    hindi_matches = find_hindi_rows_in_chunks(chunks_dir, hindi_texts)
    if not hindi_matches:
        print("No Hindi texts found in chunks. Nothing to fix.")
        return

    # Step 3: Fix chunks with corrected joy scores
    print(f"\nServer: {args.server_url}")
    total_fixed = fix_chunks(
        chunks_dir, hindi_matches, args.server_url, args.batch_size, args.dry_run
    )

    # Step 4: Rebuild final parquet
    if total_fixed > 0 and not args.dry_run:
        rebuild_final_parquet(chunks_dir, output_parquet, args.dry_run)

    print("\nDone!")


if __name__ == "__main__":
    main()
