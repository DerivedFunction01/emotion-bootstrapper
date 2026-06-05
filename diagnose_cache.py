"""
Diagnostic tool to inspect tokenized cache and identify index issues.
Run this on an existing cache to see what's wrong before inference fails.
"""

import argparse
from pathlib import Path
from collections import Counter
import json

import pandas as pd
from datasets import Dataset
from emotion_bootstrapper import VerboseSemanticBootstrapper


def load_parquet_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing cache file: {path}")
    return Dataset.from_pandas(pd.read_parquet(path), preserve_index=False)


def diagnose_cache(cache_dir: str | Path) -> None:
    """
    Inspect a tokenized cache and report any issues that would cause KeyErrors.
    """
    cache_dir = Path(cache_dir)

    print("=" * 80)
    print("EMOTION CACHE DIAGNOSTIC TOOL")
    print("=" * 80)

    # Load metadata
    meta_path = cache_dir / "cache_meta.json"
    if not meta_path.exists():
        print(f"❌ cache_meta.json not found at {meta_path}")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    print(f"\n📊 Cache Metadata:")
    print(f"  Format version: {meta.get('cache_format_version')}")
    print(f"  Dataset: {meta.get('dataset_path')} ({meta.get('dataset_config')})")
    print(f"  Model: {meta.get('model')}")
    print(f"  Num emotions: {meta.get('num_emotions')}")
    print(f"  Num rows: {meta.get('num_rows')}")

    # Load tokenized dataset
    tokenized_path = cache_dir / "tokenized.parquet"
    if not tokenized_path.exists():
        print(f"\n❌ tokenized.parquet not found at {tokenized_path}")
        return

    print(f"\n📂 Loading tokenized dataset...")
    tokenized = load_parquet_dataset(tokenized_path)
    print(f"  Loaded {len(tokenized)} tokenized examples")
    print(f"  Columns: {tokenized.column_names}")

    # Load texts dataset
    texts_path = cache_dir / "texts.parquet"
    if not texts_path.exists():
        print(f"\n⚠️  texts.parquet not found at {texts_path}")
        num_texts = None
    else:
        texts = load_parquet_dataset(texts_path)
        num_texts = len(texts)
        print(f"\n📄 Texts dataset:")
        print(f"  Total texts: {num_texts}")

    # Calculate expected structure
    num_emotions = meta.get("num_emotions", 7)  # Default to 7 emotions
    if num_texts is not None:
        expected_tokenized_rows = num_texts * num_emotions
    else:
        expected_tokenized_rows = None

    print(f"\n🔍 Index Structure Analysis:")
    print(f"  Expected emotions per text: {num_emotions}")
    if expected_tokenized_rows is not None:
        print(
            f"  Expected tokenized rows: {num_texts} texts × {num_emotions} emotions = {expected_tokenized_rows}"
        )
        print(f"  Actual tokenized rows: {len(tokenized)}")
        if len(tokenized) == expected_tokenized_rows:
            print(f"  ✓ Structure is correct")
        else:
            print(
                f"  ❌ MISMATCH! Expected {expected_tokenized_rows}, got {len(tokenized)}"
            )

    # Check text_index validity
    if "text_index" in tokenized.column_names:
        text_indices = tokenized["text_index"]
        print(f"\n🔢 Text Index Analysis:")
        print(f"  Min: {min(text_indices)}")
        print(f"  Max: {max(text_indices)}")
        print(f"  Type: {type(text_indices[0])}")

        negative_count = sum(1 for idx in text_indices if idx < 0)
        if negative_count > 0:
            print(f"  ❌ {negative_count} NEGATIVE indices (invalid!)")

        if num_texts is not None:
            out_of_range = sum(1 for idx in text_indices if idx >= num_texts)
            if out_of_range > 0:
                print(f"  ❌ {out_of_range} indices >= {num_texts} (out of range!)")

        # Check distribution
        text_index_counts = Counter(text_indices)
        print(f"  Unique text indices: {len(text_index_counts)}")
        count_distribution = Counter(text_index_counts.values())
        print(f"  Count per text index: {dict(count_distribution)}")

        if (
            len(count_distribution) == 1
            and list(count_distribution.keys())[0] == num_emotions
        ):
            print(f"  ✓ Each text has exactly {num_emotions} tokenized examples")
        else:
            print(f"  ❌ Uneven distribution detected!")
    else:
        print(f"\n❌ text_index column not found!")

    # Check label_index validity
    if "label_index" in tokenized.column_names:
        label_indices = tokenized["label_index"]
        print(f"\n🏷️  Label Index Analysis:")
        print(f"  Min: {min(label_indices)}")
        print(f"  Max: {max(label_indices)}")
        print(f"  Type: {type(label_indices[0])}")

        negative_count = sum(1 for idx in label_indices if idx < 0)
        if negative_count > 0:
            print(f"  ❌ {negative_count} NEGATIVE indices (invalid!)")

        out_of_range = sum(1 for idx in label_indices if idx >= num_emotions)
        if out_of_range > 0:
            print(f"  ❌ {out_of_range} indices >= {num_emotions} (out of range!)")

        label_index_counts = Counter(label_indices)
        print(f"  Unique label indices: {len(label_index_counts)}")
        if len(label_index_counts) == num_emotions and all(
            c == len(tokenized) // num_emotions for c in label_index_counts.values()
        ):
            print(f"  ✓ Each emotion label appears equally")
        elif set(label_index_counts.keys()) == set(range(num_emotions)):
            counts = [label_index_counts[i] for i in range(num_emotions)]
            print(f"  ⚠️  Unequal counts per emotion: {counts}")
        else:
            print(
                f"  ❌ Missing or invalid emotion indices: {sorted(label_index_counts.keys())}"
            )
    else:
        print(f"\n❌ label_index column not found!")

    # Summary
    print(f"\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)

    issues = []
    if "text_index" in tokenized.column_names and any(
        idx < 0 for idx in tokenized["text_index"]
    ):
        issues.append("Negative text indices detected")
    if "label_index" in tokenized.column_names and any(
        idx < 0 for idx in tokenized["label_index"]
    ):
        issues.append("Negative label indices detected")
    if (
        expected_tokenized_rows is not None
        and len(tokenized) != expected_tokenized_rows
    ):
        issues.append("Tokenized row count mismatch")

    if issues:
        print("\n❌ ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n  SOLUTION: Rebuild the cache using:")
        print("    python build_cache.py --cache-dir <dir> --zip-path <zip>")
        print(
            "\n  The fixed build_cache.py includes validation that will catch these issues."
        )
    else:
        print("\n✅ Cache appears to be valid!")
        print("   You can safely use run_bootstrap.py to run inference.")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose issues with an emotion tokenized cache"
    )
    parser.add_argument(
        "--cache-dir", default="./emotion_cache", help="Path to the cache directory"
    )
    args = parser.parse_args()

    diagnose_cache(args.cache_dir)
