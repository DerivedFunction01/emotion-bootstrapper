#!/usr/bin/env python3
"""
Generalized emotion fixer for any emotion/language combination.
Uses hypotheses from emotion_bootstrapper.py (SEMANTIC_HYPOTHESES_MULTILINGUAL).
Supports parallel inference across multiple servers for speed.

USAGE:
  # Fix Hindi joy
  python fix_emotion_hypotheses.py \
    --chunks-dir ./bootstrap_work_mt/chunks \
    --output-parquet ./emotion_bootstrapped_mt.parquet \
    --target-language hindi \
    --target-emotions joy \
    --server-urls http://127.0.0.1:8000 http://127.0.0.1:8001

  # Fix multiple emotions
  python fix_emotion_hypotheses.py \
    --chunks-dir ./bootstrap_work_mt/chunks \
    --output-parquet ./emotion_bootstrapped_mt.parquet \
    --target-language spanish \
    --target-emotions fear anger \
    --server-urls http://127.0.0.1:8000 http://127.0.0.1:8001

  # Dry run
  python fix_emotion_hypotheses.py \
    ... \
    --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import pandas as pd
from tqdm import tqdm

# Import hypotheses from emotion_bootstrapper
from emotion_bootstrapper import SEMANTIC_HYPOTHESES_MULTILINGUAL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix emotion hypothesis scores for any emotion/language combination."
    )
    parser.add_argument(
        "--chunks-dir", required=True, help="Directory containing chunk_*.parquet files"
    )
    parser.add_argument(
        "--output-parquet", required=True, help="Final output parquet path to patch"
    )
    parser.add_argument(
        "--target-language",
        required=True,
        help="Language (e.g., hindi, spanish, french)",
    )
    parser.add_argument(
        "--target-emotions",
        nargs="+",
        required=True,
        help="Emotion(s) to fix (e.g., joy fear anger)",
    )
    parser.add_argument(
        "--server-urls",
        nargs="+",
        required=True,
        help="Inference server URLs (e.g., http://127.0.0.1:8000 http://127.0.0.1:8001)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for inference per server"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max parallel workers (default: len(servers) * 2)",
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


def get_hypotheses_for_language(
    language: str, target_emotions: list[str]
) -> dict[str, str]:
    """Get hypotheses from SEMANTIC_HYPOTHESES_MULTILINGUAL for the target language and emotions."""
    lang_lower = language.lower()

    if lang_lower not in SEMANTIC_HYPOTHESES_MULTILINGUAL:
        available = ", ".join(SEMANTIC_HYPOTHESES_MULTILINGUAL.keys())
        raise ValueError(
            f"Language '{language}' not found in SEMANTIC_HYPOTHESES_MULTILINGUAL. "
            f"Available: {available}"
        )

    lang_hypotheses = SEMANTIC_HYPOTHESES_MULTILINGUAL[lang_lower]

    # Validate emotions
    emotion_hypotheses = {}
    for emotion in target_emotions:
        if emotion not in lang_hypotheses:
            available = ", ".join(lang_hypotheses.keys())
            raise ValueError(
                f"Emotion '{emotion}' not found for language '{language}'. "
                f"Available: {available}"
            )
        emotion_hypotheses[emotion] = lang_hypotheses[emotion]

    return emotion_hypotheses


def infer_emotion_scores(
    server_url: str,
    texts: list[str],
    emotion_name: str,
    hypothesis: str,
    batch_size: int = 32,
) -> dict[str, float]:
    """Infer emotion scores for texts using the server."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

    # Create premise/hypothesis pairs
    pair_texts = texts
    pair_hypotheses = [hypothesis] * len(texts)

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
        desc=f"Inferring {emotion_name} scores",
        total=(len(tokenized["input_ids"]) + batch_size - 1) // batch_size,
        leave=False,
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


def infer_batch_with_server(
    server_url: str,
    texts: list[str],
    emotion_name: str,
    hypothesis: str,
    batch_size: int,
) -> dict[str, float]:
    """Wrapper for parallel execution."""
    return infer_emotion_scores(server_url, texts, emotion_name, hypothesis, batch_size)


def fix_chunks(
    chunks_dir: Path,
    target_emotions: list[str],
    emotion_hypotheses: dict[str, str],
    server_urls: list[str],
    batch_size: int = 32,
    max_workers: int | None = None,
    dry_run: bool = False,
) -> int:
    """Fix emotion scores in all chunks using parallel server inference."""
    total_fixed = 0

    if max_workers is None:
        max_workers = max(1, len(server_urls) * 2)

    chunk_paths = sorted(chunks_dir.glob("chunk_*.parquet"))

    for chunk_path in tqdm(chunk_paths, desc="Processing chunks"):
        df = pd.read_parquet(chunk_path)

        # Get all texts in this chunk
        texts_in_chunk = df["text"].tolist()

        # Infer all emotions in parallel using different servers
        emotion_scores_per_text: dict[str, dict[str, float]] = {
            emotion: {} for emotion in target_emotions
        }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for emotion_idx, emotion_name in enumerate(target_emotions):
                hypothesis = emotion_hypotheses[emotion_name]
                server_url = server_urls[emotion_idx % len(server_urls)]

                future = executor.submit(
                    infer_batch_with_server,
                    server_url,
                    texts_in_chunk,
                    emotion_name,
                    hypothesis,
                    batch_size,
                )
                futures[future] = emotion_name

            # Collect results
            for future in as_completed(futures):
                emotion_name = futures[future]
                emotion_scores_per_text[emotion_name] = future.result()

        # Build new emotion_vector column as list of dicts
        emotion_vectors = []
        for idx in range(len(df)):
            old_vector = df.iloc[idx]["emotion_vector"]
            new_vector = dict(old_vector)

            # Update all target emotions
            text = df.iloc[idx]["text"]
            for emotion_name in target_emotions:
                if text in emotion_scores_per_text[emotion_name]:
                    new_vector[emotion_name] = emotion_scores_per_text[emotion_name][
                        text
                    ]

            emotion_vectors.append(new_vector)

        # Replace entire column
        df["emotion_vector"] = emotion_vectors

        if not dry_run:
            df.to_parquet(chunk_path, index=False)
            print(f"Updated {chunk_path.name}")

        total_fixed += len(df)

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

    print(f"\nRebuilding {output_parquet} from {len(chunk_paths)} chunks...")
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

    # Get hypotheses from emotion_bootstrapper
    emotion_hypotheses = get_hypotheses_for_language(
        args.target_language, args.target_emotions
    )

    print("=" * 80)
    print("EMOTION HYPOTHESIS FIXER")
    print("=" * 80)
    print(f"Target Language: {args.target_language}")
    print(f"Target Emotions: {', '.join(args.target_emotions)}")
    print(f"Hypotheses:")
    for emotion, hypothesis in emotion_hypotheses.items():
        print(f"  {emotion}: {hypothesis}")
    print(f"\nServers: {len(args.server_urls)} server(s)")
    for idx, url in enumerate(args.server_urls, 1):
        print(f"  {idx}. {url}")
    print(f"Chunks Directory: {chunks_dir}")
    print(f"Output Parquet: {output_parquet}")
    if args.dry_run:
        print("MODE: DRY RUN (no writes)")
    print("=" * 80 + "\n")

    # Fix chunks with corrected scores
    print(f"Fixing {len(args.target_emotions)} emotion(s)...\n")
    total_fixed = fix_chunks(
        chunks_dir,
        args.target_emotions,
        emotion_hypotheses,
        args.server_urls,
        args.batch_size,
        args.max_workers,
        args.dry_run,
    )

    # Rebuild final parquet
    if total_fixed > 0 and not args.dry_run:
        rebuild_final_parquet(chunks_dir, output_parquet, args.dry_run)

    print("\nDone!")


if __name__ == "__main__":
    main()
