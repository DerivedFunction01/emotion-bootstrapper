from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset

from emotion_bootstrapper import VerboseSemanticBootstrapper, save_dataset_as_parquet
from emotion_cache import load_dataset_cache, unzip_cache_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run emotion bootstrap inference from a cache zip.")
    parser.add_argument("--cache-zip", required=True)
    parser.add_argument("--work-dir", default="./bootstrap_work")
    parser.add_argument("--output-path", default="./emotion_bootstrapped.parquet")
    parser.add_argument("--stats-path", default="./emotion_bootstrap_stats.json")
    parser.add_argument("--model", default="facebook/bart-large-mnli")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-proc", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    cache_dir = work_dir / "cache"
    unzip_cache_dir(args.cache_zip, cache_dir)

    tokenized = load_dataset_cache(cache_dir)
    if not isinstance(tokenized, Dataset):
        raise TypeError("Expected a single Dataset cache for bootstrap inference")

    bootstrapper = VerboseSemanticBootstrapper(model=args.model)
    required = {"text_index", "label_index", "source_text"}
    if not required.issubset(set(tokenized.column_names)):
        raise ValueError("Tokenized cache is missing required index columns")

    text_count = max(tokenized["text_index"]) + 1 if len(tokenized) else 0
    texts = [""] * text_count
    for text_index, source_text in zip(tokenized["text_index"], tokenized["source_text"]):
        if not texts[text_index]:
            texts[text_index] = source_text
    dataset = Dataset.from_dict({"text": texts})

    text_to_scores = {}
    pair_batch_size = max(1, args.batch_size * len(bootstrapper.hypotheses))
    for start in range(0, len(tokenized), pair_batch_size):
        batch = tokenized[start : start + pair_batch_size]
        inputs = {
            key: torch.tensor(value).to(bootstrapper.device)
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.no_grad():
            logits = bootstrapper.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            entailment_scores = probs[:, bootstrapper.entailment_id].detach().cpu().tolist()
        for text_index, label_index, score in zip(
            batch["text_index"], batch["label_index"], entailment_scores
        ):
            text_to_scores.setdefault(text_index, [0.0] * len(bootstrapper.hypotheses))
            text_to_scores[text_index][label_index] = float(score)

    emotion_vectors = [
        {
            emotion: float(score)
            for emotion, score in zip(bootstrapper.emotion_labels, text_to_scores[i])
        }
        for i in range(len(dataset))
    ]
    bootstrapped = dataset.add_column("emotion_vector", emotion_vectors)
    save_dataset_as_parquet(bootstrapped, args.output_path)
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    main()
