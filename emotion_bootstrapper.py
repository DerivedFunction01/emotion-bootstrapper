import json
import math
import os
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SEMANTIC_HYPOTHESES = {
    "fear": "Someone has at least at least a strong feeling of fear caused by the threat of danger or pain",
    "anger": "Someone has at least a strong feeling of annoyance, displeasure, or hostility",
    "surprise": "Someone has at least a strong feeling of mild shock or astonishment",
    "joy": "Someone has at least a strong feeling of great pleasure and happiness",
    "sadness": "Someone has at least a strong feeling of deep distress caused by loss or disappointment",
    "disgust": "Someone has at least a strong feeling of revulsion or strong disapproval",
    "urgency": "Someone has at least a strong feeling of urgency to act immediately due to time pressure",
    "confusion": "Someone has at least a strong feeling of struggling to understand something complex or confusing",
    "emotion": "There is someone with feelings of either fear, anger, surprise, joy, sadness, disgust, urgency, or confusion",
}


def load_primary_dataset(dataset_path: str, dataset_config: str) -> Dataset:
    """Load the most relevant split from a Hugging Face dataset."""
    dataset_dict = load_dataset(dataset_path, dataset_config)

    if "train" in dataset_dict:
        return dataset_dict["train"]

    first_split_name = next(iter(dataset_dict.keys()))
    return dataset_dict[first_split_name]


def save_dataset_as_parquet(dataset: Dataset, output_path: str) -> None:
    """Persist a Hugging Face Dataset to parquet."""
    df = dataset.to_pandas()
    df.to_parquet(output_path, index=False)


def load_parquet_size_mb(output_path: str) -> float:
    """Read parquet back so we can report file size / memory usage."""
    return float(pd.read_parquet(output_path).memory_usage(deep=True).sum() / 1e6)


def write_json_atomic(path: str, payload: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def chunked(items: List[str], size: int) -> List[List[str]]:
    """Yield consecutive chunks from a list."""
    if size <= 0:
        raise ValueError("batch_size must be a positive integer")
    return [items[i : i + size] for i in range(0, len(items), size)]


def zip_directory(source_dir: str, zip_path: str) -> str:
    """Zip a directory recursively."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                arcname = os.path.relpath(file_path, source_dir)
                zf.write(file_path, arcname)
    return zip_path


class VerboseSemanticBootstrapper:
    """
    Bootstrap emotion labels using verbose semantic hypotheses.
    Optimized for high-entropy emotional text and negation sensitivity.
    """

    def __init__(
        self, model: str = "facebook/bart-large-mnli", device_map: str = "auto"
    ):
        print(f"Loading model/tokenizer: {model}")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device_map != "cpu" else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()
        self.emotion_labels = list(SEMANTIC_HYPOTHESES.keys())
        self.hypotheses = list(SEMANTIC_HYPOTHESES.values())
        self.entailment_id = self._find_entailment_id()

    def _find_entailment_id(self) -> int:
        label2id = getattr(self.model.config, "label2id", {}) or {}
        for key, idx in label2id.items():
            if str(key).lower() == "entailment":
                return int(idx)
        return 2

    def label_text(self, text: str) -> Dict[str, float]:
        return self.label_texts([text])[0]

    def _tokenize_pairs(
        self, batch: Dict[str, List[str]], indices: List[int], text_column: str
    ) -> Dict[str, List]:
        texts = batch[text_column]
        input_texts = []
        hypotheses = []
        text_indices = []
        label_indices = []

        for text_index, text in enumerate(texts):
            global_text_index = int(indices[text_index])
            for label_index, hypothesis in enumerate(self.hypotheses):
                input_texts.append(text)
                hypotheses.append(hypothesis)
                text_indices.append(global_text_index)
                label_indices.append(label_index)

        tokenized = self.tokenizer(
            input_texts,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
        )
        tokenized["text_index"] = text_indices
        tokenized["label_index"] = label_indices
        return tokenized

    def tokenize_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        batch_size: int = 1000,
        num_proc: int | None = None,
        cache_dir: str | None = None,
    ) -> Dataset:
        print(
            f"Tokenizing {len(dataset)} rows into premise/hypothesis pairs "
            f"with num_proc={num_proc or 1}..."
        )

        tokenized = dataset.map(
            lambda batch, indices: self._tokenize_pairs(batch, indices, text_column),
            batched=True,
            with_indices=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc="tokenizing",
        )
        if cache_dir:
            tokenized.save_to_disk(cache_dir)
        return tokenized

    def label_texts(self, texts: List[str], inference_batch_size: int = 32) -> List[Dict[str, float]]:
        if not texts:
            return []

        pair_texts = []
        pair_hypotheses = []
        for text in texts:
            for hypothesis in self.hypotheses:
                pair_texts.append(text)
                pair_hypotheses.append(hypothesis)

        all_entailment_scores: List[float] = []
        for start in tqdm(
            range(0, len(pair_texts), inference_batch_size),
            total=math.ceil(len(pair_texts) / inference_batch_size),
            desc="inference",
        ):
            end = start + inference_batch_size
            inputs = self.tokenizer(
                pair_texts[start:end],
                pair_hypotheses[start:end],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                all_entailment_scores.extend(
                    probs[:, self.entailment_id].detach().cpu().tolist()
                )

        grouped = [
            all_entailment_scores[i : i + len(self.hypotheses)]
            for i in range(0, len(all_entailment_scores), len(self.hypotheses))
        ]
        return [
            {
                emotion: float(score)
                for emotion, score in zip(self.emotion_labels, scores)
            }
            for scores in grouped
        ]

    def bootstrap_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        batch_size: int = 32,
        show_progress: bool = True,
        num_proc: int | None = None,
        tokenized_cache_path: str | None = None,
        tokenized_zip_path: str | None = None,
        raw_cache_path: str | None = None,
    ) -> Dataset:
        if raw_cache_path:
            os.makedirs(raw_cache_path, exist_ok=True)
            Dataset.from_dict({"text": dataset[text_column]}).to_parquet(
                os.path.join(raw_cache_path, "texts.parquet")
            )
            write_json_atomic(
                os.path.join(raw_cache_path, "cache_meta.json"),
                {
                    "dataset_rows": len(dataset),
                    "text_column": text_column,
                    "num_emotions": len(self.hypotheses),
                    "batch_size": batch_size,
                },
            )

        tokenized_dataset = self.tokenize_dataset(
            dataset,
            text_column=text_column,
            batch_size=batch_size,
            num_proc=num_proc,
            cache_dir=tokenized_cache_path,
        )
        if tokenized_cache_path and tokenized_zip_path:
            if os.path.exists(tokenized_zip_path):
                raise FileExistsError(f"Refusing to overwrite existing zip: {tokenized_zip_path}")
            zip_directory(tokenized_cache_path, tokenized_zip_path)

        text_to_scores: Dict[int, List[float]] = {}
        pair_batch_size = max(1, batch_size * len(self.hypotheses))

        for start in tqdm(
            range(0, len(tokenized_dataset), pair_batch_size),
            total=math.ceil(len(tokenized_dataset) / pair_batch_size),
            disable=not show_progress,
            desc="bootstrapping",
        ):
            batch = tokenized_dataset[start : start + pair_batch_size]
            inputs = {
                key: torch.tensor(value).to(self.device)
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                entailment_scores = probs[:, self.entailment_id].detach().cpu().tolist()

            for text_index, label_index, score in zip(
                batch["text_index"], batch["label_index"], entailment_scores
            ):
                text_to_scores.setdefault(text_index, [0.0] * len(self.hypotheses))
                text_to_scores[text_index][label_index] = float(score)

        emotion_vectors = [
            {
                emotion: float(score)
                for emotion, score in zip(self.emotion_labels, text_to_scores[i])
            }
            for i in range(len(dataset))
        ]

        return dataset.add_column("emotion_vector", emotion_vectors)

    def get_statistics(self, dataset: Dataset) -> Dict:
        emotion_vectors = dataset["emotion_vector"]

        emotion_stats = {}
        for emotion in self.emotion_labels:
            scores = [ev[emotion] for ev in emotion_vectors]
            emotion_stats[emotion] = {
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "count_above_0.5": sum(1 for s in scores if s > 0.5),
                "count_above_0.7": sum(1 for s in scores if s > 0.7),
            }

        num_emotions_per_text = [
            sum(1 for score in ev.values() if score > 0.5) for ev in emotion_vectors
        ]

        return {
            "total_texts": len(emotion_vectors),
            "emotion_statistics": emotion_stats,
            "multi_emotion_distribution": {
                "mean_emotions_per_text": float(np.mean(num_emotions_per_text)),
                "median_emotions_per_text": float(np.median(num_emotions_per_text)),
                "max_emotions_per_text": int(np.max(num_emotions_per_text)),
                "texts_with_single_emotion": sum(1 for n in num_emotions_per_text if n == 1),
                "texts_with_multiple_emotions": sum(1 for n in num_emotions_per_text if n > 1),
                "texts_with_no_clear_emotion": sum(1 for n in num_emotions_per_text if n == 0),
            },
        }

    def print_statistics(self, stats: Dict) -> None:
        print("\n" + "=" * 80)
        print("BOOTSTRAPPED DATASET STATISTICS")
        print("=" * 80)
        print(f"\nTotal texts: {stats['total_texts']}")

        print("\nEmotion Statistics:")
        print("-" * 80)
        print(
            f"{'Emotion':<12} {'Mean':<8} {'Median':<8} {'Std':<8} {'>0.5':<8} {'>0.7':<8}"
        )
        print("-" * 80)
        for emotion, stats_dict in stats["emotion_statistics"].items():
            print(
                f"{emotion:<12} {stats_dict['mean']:<8.3f} {stats_dict['median']:<8.3f} "
                f"{stats_dict['std']:<8.3f} {stats_dict['count_above_0.5']:<8} "
                f"{stats_dict['count_above_0.7']:<8}"
            )

        print("\nMulti-Emotion Distribution:")
        print("-" * 80)
        me_dist = stats["multi_emotion_distribution"]
        print(f"Average emotions per text: {me_dist['mean_emotions_per_text']:.2f}")
        print(f"Median emotions per text: {me_dist['median_emotions_per_text']:.1f}")
        print(f"Max emotions in single text: {me_dist['max_emotions_per_text']}")
        print(f"Texts with single clear emotion: {me_dist['texts_with_single_emotion']}")
        print(f"Texts with multiple emotions: {me_dist['texts_with_multiple_emotions']}")
        print(f"Texts with no clear emotion: {me_dist['texts_with_no_clear_emotion']}")
        print("=" * 80 + "\n")