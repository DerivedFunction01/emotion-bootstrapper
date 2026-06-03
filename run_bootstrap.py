from __future__ import annotations

import argparse
import math
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from datasets import Dataset

from emotion_bootstrapper import VerboseSemanticBootstrapper, save_dataset_as_parquet
from emotion_cache import load_parquet_dataset, unzip_cache_dir


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


def _infer_shard(shard_payload: dict) -> dict[int, list[float]]:
    device_id = shard_payload["device_id"]
    model_name = shard_payload["model"]
    batch_size = shard_payload["batch_size"]
    text_indices = shard_payload["text_indices"]
    label_indices = shard_payload["label_indices"]
    input_ids = shard_payload["input_ids"]
    attention_mask = shard_payload["attention_mask"]
    token_type_ids = shard_payload.get("token_type_ids")

    bootstrapper = VerboseSemanticBootstrapper(model=model_name, device_map="cpu")
    if torch.cuda.is_available():
        bootstrapper.device = torch.device(f"cuda:{device_id}")
        bootstrapper.model.to(bootstrapper.device)
    bootstrapper.model.eval()

    results: dict[int, list[float]] = {}
    pair_batch_size = max(1, batch_size * len(bootstrapper.hypotheses))

    for start in range(0, len(text_indices), pair_batch_size):
        end = start + pair_batch_size
        inputs = {
            "input_ids": torch.tensor(input_ids[start:end]).to(bootstrapper.device),
            "attention_mask": torch.tensor(attention_mask[start:end]).to(bootstrapper.device),
        }
        if token_type_ids is not None:
            inputs["token_type_ids"] = torch.tensor(token_type_ids[start:end]).to(
                bootstrapper.device
            )
        with torch.no_grad():
            logits = bootstrapper.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            entailment_scores = probs[:, bootstrapper.entailment_id].detach().cpu().tolist()

        for text_index, label_index, score in zip(
            text_indices[start:end], label_indices[start:end], entailment_scores
        ):
            results.setdefault(text_index, [0.0] * len(bootstrapper.hypotheses))
            results[text_index][label_index] = float(score)

    return results


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    cache_dir = work_dir / "cache"
    unzip_cache_dir(args.cache_zip, cache_dir)

    tokenized = load_parquet_dataset(cache_dir / "tokenized.parquet")
    texts_ds = load_parquet_dataset(cache_dir / "texts.parquet")

    bootstrapper = VerboseSemanticBootstrapper(model=args.model)
    required = {"text_index", "label_index", "input_ids", "attention_mask"}
    if not required.issubset(set(tokenized.column_names)):
        raise ValueError("Tokenized cache is missing required index columns")

    dataset = Dataset.from_dict({"text": texts_ds["text"]})
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Detected {gpu_count} GPU(s)")

    if gpu_count <= 1:
        shards = [tokenized]
    else:
        shard_size = math.ceil(len(tokenized) / gpu_count)
        shards = [
            tokenized.select(range(start, min(start + shard_size, len(tokenized))))
            for start in range(0, len(tokenized), shard_size)
        ]

    shard_payloads = []
    for device_id, shard in enumerate(shards):
        shard_payloads.append(
            {
                "device_id": device_id,
                "model": args.model,
                "batch_size": args.batch_size,
                "text_indices": shard["text_index"],
                "label_indices": shard["label_index"],
                "input_ids": shard["input_ids"],
                "attention_mask": shard["attention_mask"],
                "token_type_ids": shard["token_type_ids"] if "token_type_ids" in shard.column_names else None,
            }
        )

    text_to_scores = {}
    if gpu_count <= 1:
        shard_results = [_infer_shard(shard_payloads[0])]
    else:
        with ProcessPoolExecutor(max_workers=gpu_count, mp_context=mp.get_context("spawn")) as executor:
            shard_results = list(executor.map(_infer_shard, shard_payloads))

    for shard_result in shard_results:
        for text_index, scores in shard_result.items():
            text_to_scores[text_index] = scores

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
