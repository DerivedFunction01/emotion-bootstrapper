from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset
from tqdm import tqdm

from emotion_bootstrapper import SEMANTIC_HYPOTHESES, VerboseSemanticBootstrapper
from emotion_cache import CACHE_FORMAT_VERSION, save_dataset_cache, zip_cache_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tokenized bootstrap cache locally.")
    parser.add_argument("--dataset-path", default="dair-ai/emotion")
    parser.add_argument("--dataset-config", default="unsplit")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--cache-dir", default="./emotion_cache")
    parser.add_argument("--zip-path", default="./emotion_cache.zip")
    parser.add_argument("--model", default="facebook/bart-large-mnli")
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument("--tokenize-batch-size", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dict = load_dataset(args.dataset_path, args.dataset_config)
    dataset = dataset_dict["train"] if "train" in dataset_dict else next(iter(dataset_dict.values()))

    bootstrapper = VerboseSemanticBootstrapper(model=args.model)
    tokenized = bootstrapper.tokenize_dataset(
        dataset,
        text_column=args.text_column,
        batch_size=args.tokenize_batch_size,
        num_proc=args.num_proc,
    )
    keep_columns = [
        column
        for column in ("input_ids", "attention_mask", "token_type_ids", "text_index", "label_index")
        if column in tokenized.column_names
    ]
    tokenized = tokenized.remove_columns(
        [column for column in tokenized.column_names if column not in keep_columns]
    )

    save_dataset_cache(
        tokenized,
        args.cache_dir,
        meta={
            "cache_format_version": CACHE_FORMAT_VERSION,
            "dataset_path": args.dataset_path,
            "dataset_config": args.dataset_config,
            "text_column": args.text_column,
            "model": args.model,
            "num_emotions": len(SEMANTIC_HYPOTHESES),
            "num_rows": len(dataset),
            "kept_columns": keep_columns,
        },
    )
    Dataset.from_dict({"text": dataset[args.text_column]}).to_parquet(
        Path(args.cache_dir) / "texts.parquet"
    )
    zip_cache_dir(args.cache_dir, args.zip_path)
    print(f"Wrote cache: {args.cache_dir}")
    print(f"Wrote zip: {args.zip_path}")


if __name__ == "__main__":
    main()
