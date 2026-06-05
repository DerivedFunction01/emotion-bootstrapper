from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

from emotion_bootstrapper import SEMANTIC_HYPOTHESES, save_dataset_as_parquet
from emotion_cache import load_json, load_parquet_dataset, unzip_cache_dir

DEFAULT_REGISTRY_PATH = "server_cluster.json"


@dataclass(frozen=True)
class InferenceServer:
    name: str
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run emotion bootstrap inference against remote GPU servers."
    )
    parser.add_argument("--cache-zip", required=True)
    parser.add_argument("--work-dir", default="./bootstrap_work")
    parser.add_argument("--output-path", default="./emotion_bootstrapped.parquet")
    parser.add_argument("--stats-path", default="./emotion_bootstrap_stats.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--flush-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--server-url",
        action="append",
        help="Inference server URL. Pass once per GPU server.",
    )
    parser.add_argument(
        "--server-registry",
        default=DEFAULT_REGISTRY_PATH,
        help="JSON file written by server_manager.py start.",
    )
    parser.add_argument(
        "--diagnose-resume",
        action="store_true",
        help="Print resume diagnostics without starting inference.",
    )
    return parser.parse_args()


def _post_json(url: str, payload: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
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
        raise RuntimeError(f"Server request failed for {url}: {exc.code} {detail}") from exc


def _build_batch_payload(batch: Dataset) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for i in range(len(batch)):
        example: dict[str, Any] = {
            "input_ids": batch["input_ids"][i],
            "attention_mask": batch["attention_mask"][i],
            "text_index": int(batch["text_index"][i]),
            "label_index": int(batch["label_index"][i]),
        }
        if "token_type_ids" in batch.column_names:
            example["token_type_ids"] = batch["token_type_ids"][i]
        examples.append(example)
    return examples


def _infer_batch(server: InferenceServer, batch: Dataset) -> list[float]:
    payload = {"examples": _build_batch_payload(batch)}
    response = _post_json(f"{server.url.rstrip('/')}/infer", payload)
    scores = response.get("entailment_scores")
    if not isinstance(scores, list):
        raise ValueError(f"Server {server.name} returned an invalid response: {response}")
    return [float(score) for score in scores]


def _build_batch_rows(
    server: InferenceServer,
    batch: Dataset,
    batch_index: int,
    start_text: int,
    end_text: int,
    texts_ds: Dataset,
    emotion_labels: list[str],
    num_hypotheses: int,
) -> tuple[int, list[dict[str, Any]]]:
    entailment_scores = _infer_batch(server, batch)

    current_scores: dict[int, list[float]] = {
        local_idx: [0.0] * num_hypotheses for local_idx in range(end_text - start_text)
    }
    for example_idx, (label_index, score) in enumerate(
        zip(batch["label_index"], entailment_scores)
    ):
        local_text_index = example_idx // num_hypotheses
        current_scores[local_text_index][int(label_index)] = float(score)

    rows = []
    for local_idx, row_idx in enumerate(range(start_text, end_text)):
        rows.append(
            {
                "text": texts_ds["text"][row_idx],
                "emotion_vector": {
                    emotion: score
                    for emotion, score in zip(emotion_labels, current_scores[local_idx])
                },
            }
        )
    return batch_index, rows


def _atomic_write_dataset_parquet(dataset: Dataset, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    dataset.to_parquet(tmp_path)
    os.replace(tmp_path, path)


def _load_servers(args: argparse.Namespace) -> list[InferenceServer]:
    if args.server_url:
        return [InferenceServer(name=f"server-{i}", url=url) for i, url in enumerate(args.server_url)]

    registry_path = Path(args.server_registry)
    registry = load_json(registry_path)
    servers = registry.get("servers", [])
    if not servers:
        raise ValueError(f"No servers found in registry {registry_path}")
    return [
        InferenceServer(name=server["name"], url=server["url"])
        for server in servers
    ]


def _print_resume_diagnostics(work_dir: Path, cache_dir: Path) -> None:
    chunk_dir = work_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    existing_chunks = sorted(chunk_dir.glob("chunk_*.parquet"))
    resume_text_index = 0
    for chunk_path in existing_chunks:
        resume_text_index += len(load_parquet_dataset(chunk_path))

    print("Resume diagnostics")
    print(f"  work_dir: {work_dir}")
    print(f"  cache_dir: {cache_dir}")
    print(f"  chunk_files: {len(existing_chunks)}")
    print(f"  completed_text_rows: {resume_text_index}")
    if (cache_dir / "cache_meta.json").exists():
        meta = load_json(cache_dir / "cache_meta.json")
        print(f"  cache_meta num_rows: {meta.get('num_rows')}")
        print(f"  cache_meta num_hypotheses: {meta.get('num_hypotheses')}")
    else:
        print("  cache_meta.json: missing")
    if (cache_dir / "tokenized.parquet").exists():
        tokenized = load_parquet_dataset(cache_dir / "tokenized.parquet")
        print(f"  tokenized_rows: {len(tokenized)}")
        print(f"  tokenized_label_range: {min(tokenized['label_index']) if len(tokenized) else 'n/a'}..{max(tokenized['label_index']) if len(tokenized) else 'n/a'}")
    else:
        print("  tokenized.parquet: missing")


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    cache_dir = work_dir / "cache"

    if args.diagnose_resume:
        _print_resume_diagnostics(work_dir, cache_dir)
        return

    cache_ready = (
        (cache_dir / "tokenized.parquet").exists()
        and (cache_dir / "texts.parquet").exists()
        and (cache_dir / "cache_meta.json").exists()
    )
    if not cache_ready:
        unzip_cache_dir(args.cache_zip, cache_dir)
    else:
        print(f"Using existing unpacked cache in {cache_dir}")

    tokenized = load_parquet_dataset(cache_dir / "tokenized.parquet")
    texts_ds = load_parquet_dataset(cache_dir / "texts.parquet")
    cache_meta = load_json(cache_dir / "cache_meta.json")
    cache_model = cache_meta.get("model")
    if not cache_model:
        raise ValueError("cache_meta.json does not declare a model")
    if args.model is not None and args.model != cache_model:
        raise ValueError(
            f"Model override {args.model!r} does not match cache model {cache_model!r}"
        )

    required = {"text_index", "label_index", "input_ids", "attention_mask"}
    if not required.issubset(set(tokenized.column_names)):
        raise ValueError("Tokenized cache is missing required index columns")

    emotion_labels = list(SEMANTIC_HYPOTHESES.keys())
    num_hypotheses = len(emotion_labels)
    dataset = Dataset.from_dict({"text": texts_ds["text"]})
    total_texts = len(dataset)
    batch_size = max(1, args.batch_size)
    servers = _load_servers(args)

    chunk_dir = work_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    existing_chunks = sorted(chunk_dir.glob("chunk_*.parquet"))
    resume_text_index = 0
    for chunk_path in existing_chunks:
        resume_text_index += len(load_parquet_dataset(chunk_path))

    if existing_chunks:
        print(
            f"Resuming from existing chunks: {len(existing_chunks)} file(s), "
            f"{resume_text_index} text row(s) already completed"
        )
    batches = list(range(resume_text_index, total_texts, batch_size))

    print(f"Loaded {total_texts} texts")
    print(f"Using {len(servers)} server(s)")
    for server in servers:
        print(f"  {server.name}: {server.url}")

    buffer: list[dict[str, Any]] = []
    chunk_index = len(existing_chunks)
    server_index = 0

    batch_results: list[list[dict[str, Any]] | None] = [None] * len(batches)
    next_to_emit = 0
    pbar = tqdm(total=len(batches), desc="Inference batches")
    max_workers = max(1, min(64, len(servers) * 2))
    executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit_batch(batch_idx: int) -> None:
        nonlocal server_index
        start_text = batches[batch_idx]
        end_text = min(start_text + batch_size, total_texts)
        batch = tokenized.select(range(start_text * num_hypotheses, end_text * num_hypotheses))
        future = executor.submit(
            _build_batch_rows,
            servers[server_index],
            batch,
            batch_idx,
            start_text,
            end_text,
            texts_ds,
            emotion_labels,
            num_hypotheses,
        )
        futures[future] = batch_idx
        server_index = (server_index + 1) % len(servers)

    futures: dict[Any, int] = {}
    next_batch_idx = 0
    for _ in range(min(len(batches), len(servers))):
        submit_batch(next_batch_idx)
        next_batch_idx += 1

    while futures:
        for future in list(as_completed(futures)):
            batch_idx = futures.pop(future)
            _, rows = future.result()
            batch_results[batch_idx] = rows
            pbar.update(1)

            while next_to_emit < len(batch_results):
                current_rows = batch_results[next_to_emit]
                if current_rows is None:
                    break
                buffer.extend(current_rows)
                batch_results[next_to_emit] = None
                next_to_emit += 1
                if len(buffer) >= args.flush_every or next_to_emit >= len(batches):
                    chunk_path = chunk_dir / f"chunk_{chunk_index:06d}.parquet"
                    _atomic_write_dataset_parquet(Dataset.from_list(buffer), chunk_path)
                    buffer.clear()
                    chunk_index += 1

            if next_batch_idx < len(batches):
                submit_batch(next_batch_idx)
                next_batch_idx += 1

    executor.shutdown(wait=True)
    pbar.close()

    if buffer:
        chunk_path = chunk_dir / f"chunk_{chunk_index:06d}.parquet"
        _atomic_write_dataset_parquet(Dataset.from_list(buffer), chunk_path)

    shard_outputs = []
    chunk_paths = sorted(chunk_dir.glob("*.parquet"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk parquet files found in {chunk_dir}")
    shard_outputs.append(concatenate_datasets([load_parquet_dataset(path) for path in chunk_paths]))

    merged = concatenate_datasets(shard_outputs)
    save_dataset_as_parquet(merged, args.output_path)
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    main()
