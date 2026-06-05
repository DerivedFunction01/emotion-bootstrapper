from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset
from tqdm import tqdm

from emotion_bootstrapper import SEMANTIC_HYPOTHESES, VerboseSemanticBootstrapper
from emotion_cache import CACHE_FORMAT_VERSION, write_json_atomic, zip_cache_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a tokenized bootstrap cache locally."
    )
    parser.add_argument(
        "--dataset-path", default="DerivedFunction01/dair-ai_emotions_sample"
    )
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--cache-dir", default="./emotion_cache")
    parser.add_argument("--zip-path", default="./emotion_cache.zip")
    parser.add_argument("--model", default="facebook/bart-large-mnli")
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument("--tokenize-batch-size", type=int, default=1000)
    return parser.parse_args()


def validate_and_fix_indices(dataset: Dataset, num_hypotheses: int) -> Dataset:
    """
    Validate that text_index and label_index are correct.
    Fix any issues that would cause KeyErrors during inference.
    """
    print("Validating indices...")

    # Check for invalid indices
    invalid_count = 0
    text_indices = dataset["text_index"]
    label_indices = dataset["label_index"]

    for i, (text_idx, label_idx) in enumerate(zip(text_indices, label_indices)):
        if not isinstance(text_idx, int) or text_idx < 0:
            print(f"  WARNING: Invalid text_index at position {i}: {text_idx}")
            invalid_count += 1
        if (
            not isinstance(label_idx, int)
            or label_idx < 0
            or label_idx >= num_hypotheses
        ):
            print(
                f"  WARNING: Invalid label_index at position {i}: {label_idx} (max should be {num_hypotheses - 1})"
            )
            invalid_count += 1

    if invalid_count > 0:
        print(f"  Found {invalid_count} invalid index entries!")
        raise ValueError(
            f"Dataset contains {invalid_count} rows with invalid indices. "
            "This will cause KeyErrors during inference. "
            "Please check the tokenization process."
        )

    print("  ✓ All indices are valid")
    return dataset


def main() -> None:
    args = parse_args()
    dataset_dict = load_dataset(args.dataset_path, args.dataset_config)
    dataset = (
        dataset_dict["train"]
        if "train" in dataset_dict
        else next(iter(dataset_dict.values()))
    )

    bootstrapper = VerboseSemanticBootstrapper(model=args.model)
    tokenized = bootstrapper.tokenize_dataset(
        dataset,
        text_column=args.text_column,
        batch_size=args.tokenize_batch_size,
        num_proc=args.num_proc,
    )

    # Validate indices before keeping columns
    num_hypotheses = len(bootstrapper.hypotheses)
    tokenized = validate_and_fix_indices(tokenized, num_hypotheses)

    keep_columns = [
        column
        for column in (
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "text_index",
            "label_index",
        )
        if column in tokenized.column_names
    ]
    tokenized = tokenized.remove_columns(
        [column for column in tokenized.column_names if column not in keep_columns]
    )

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenized.to_parquet(cache_dir / "tokenized.parquet")
    Dataset.from_dict({"text": dataset[args.text_column]}).to_parquet(
        cache_dir / "texts.parquet"
    )
    write_json_atomic(
        cache_dir / "cache_meta.json",
        {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "dataset_path": args.dataset_path,
            "dataset_config": args.dataset_config,
            "text_column": args.text_column,
            "model": args.model,
            "num_emotions": len(SEMANTIC_HYPOTHESES),
            "num_rows": len(dataset),
            "num_hypotheses": num_hypotheses,
            "kept_columns": keep_columns,
        },
    )
    zip_cache_dir(args.cache_dir, args.zip_path)
    print(f"Wrote cache: {args.cache_dir}")
    print(f"Wrote zip: {args.zip_path}")


if __name__ == "__main__":
    main()
