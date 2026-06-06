# %%
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
EMOTIONS_PARQUET_URL = (
    "https://huggingface.co/datasets/DerivedFunction01/emotions-zero-shot/resolve/main/"
    "emotion_bootstrapped.parquet"
)
URGENCY_PARQUET_URL = (
    "https://huggingface.co/datasets/DerivedFunction01/emotions-zero-shot/resolve/main/"
    "emotion_bootstrapped_urgency.parquet"
)
ARXIV_PARQUET_URL = (
    "https://huggingface.co/datasets/DerivedFunction01/emotions-zero-shot/resolve/main/"
    "emotion_bootstrapped_arxiv.parquet"
)

OUTPUT_PATH = Path("emotions_decayed_samples.parquet")
AUGMENTED_PARQUET_URL = (
    "https://huggingface.co/datasets/emotions-entailment/zero-shot-emotions-mt/resolve/main/"
    "emotion_translation_augmented_filtered.parquet"
)


def local_path_from_url(url: str) -> Path:
    return Path(Path(urlparse(url).path).name)


LOCAL_EMOTIONS_PARQUET = local_path_from_url(EMOTIONS_PARQUET_URL)
LOCAL_URGENCY_PARQUET = local_path_from_url(URGENCY_PARQUET_URL)
LOCAL_ARXIV_PARQUET = local_path_from_url(ARXIV_PARQUET_URL)
LOCAL_AUGMENTED_PARQUET = local_path_from_url(AUGMENTED_PARQUET_URL)


# %%
def load_parquet_frame(
    local_path: Path, remote_url: str, *, allow_missing: bool = False
) -> pd.DataFrame:
    if local_path.exists():
        print(f"Loaded {local_path} from local file system.")
        return pd.read_parquet(local_path)

    print(
        f"Local file {local_path} not found. Attempting to download from {remote_url} ..."
    )
    try:
        frame = pd.read_parquet(remote_url)
        frame.to_parquet(local_path)
        print(f"Downloaded and saved {local_path}.")
        return frame
    except Exception as exc:
        if allow_missing:
            print(f"Skipping {local_path}: {exc}")
            return pd.DataFrame()
        raise


def load_local_augmented_frame(local_path: Path) -> pd.DataFrame:
    if not local_path.exists():
        print(f"No augmented parquet found at {local_path}; continuing without it.")
        return pd.DataFrame()

    print(f"Loaded augmented parquet from {local_path}.")
    return pd.read_parquet(local_path)


def extract_emotions(
    df: pd.DataFrame, vector_col: str = "emotion_vector"
) -> list[dict]:
    vectors: list[dict] = []
    for _, row in df.iterrows():
        value = row[vector_col]
        if isinstance(value, dict):
            vectors.append(value)
        elif isinstance(value, str):
            vectors.append(json.loads(value))
    return vectors


def compute_emotion_stats(
    vectors: list[dict], dataset_name: str = "Dataset"
) -> dict[str, dict]:
    if not vectors:
        print(f"{dataset_name}: No vectors found")
        return {}

    stats: dict[str, dict] = {}
    for emotion in vectors[0].keys():
        scores = [v[emotion] for v in vectors if emotion in v]
        nonzero = sum(1 for s in scores if s > 0.0)
        stats[emotion] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
            "count_nonzero": int(nonzero),
            "pct_nonzero": float(100 * nonzero / len(scores)),
        }
    return stats


def compute_top_rank_averages(vectors: list[dict]) -> dict[int, dict]:
    if not vectors:
        return {}

    rank_scores: dict[int, list[float]] = defaultdict(list)
    for vector in vectors:
        for rank, (_, score) in enumerate(
            sorted(vector.items(), key=lambda item: item[1], reverse=True), start=1
        ):
            rank_scores[rank].append(score)

    return {
        rank: {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "median": float(np.median(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "count": len(scores),
        }
        for rank, scores in sorted(rank_scores.items())
    }


def compute_emotion_count_distribution(
    vectors: list[dict], thresholds: list[float] | None = None
) -> dict[float, list[int]]:
    thresholds = thresholds or [0.3, 0.5, 0.7]
    distributions = {threshold: [] for threshold in thresholds}
    for vector in vectors:
        for threshold in thresholds:
            distributions[threshold].append(
                sum(1 for score in vector.values() if score > threshold)
            )
    return distributions


def print_emotion_stats(title: str, stats: dict[str, dict]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for emotion in sorted(stats.keys(), key=lambda e: stats[e]["mean"], reverse=True):
        s = stats[emotion]
        print(f"\n{emotion.upper()}")
        print(f"  Mean:     {s['mean']:.4f}")
        print(f"  Std:      {s['std']:.4f}")
        print(f"  Median:   {s['median']:.4f}")
        print(f"  Min/Max:  {s['min']:.4f} / {s['max']:.4f}")
        print(f"  Non-zero: {s['count_nonzero']:,} ({s['pct_nonzero']:.1f}%)")


def print_rank_stats(title: str, rank_stats: dict[int, dict], top_n: int = 9) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for rank in sorted(rank_stats.keys())[:top_n]:
        s = rank_stats[rank]
        print(
            f"Rank {rank}: mean={s['mean']:.4f}, median={s['median']:.4f}, std={s['std']:.4f}"
        )


def print_count_distribution(title: str, distributions: dict[float, list[int]]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for threshold in sorted(distributions.keys()):
        counts = distributions[threshold]
        unique, unique_counts = np.unique(counts, return_counts=True)
        print(f"\nEmotions scoring > {threshold}:")
        print(f"  Mean:     {np.mean(counts):.2f}")
        print(f"  Std:      {np.std(counts):.2f}")
        print(f"  Median:   {np.median(counts):.0f}")
        print(f"  Min/Max:  {np.min(counts)} / {np.max(counts)}")
        print("  Distribution: ", end="")
        for val, cnt in zip(unique, unique_counts):
            print(f"{int(val)}({cnt // len(counts) * 100:.0f}%) ", end="")
        print()


# %%
def decay_multiplier(
    rank: int,
    start_penalty: float,
    increment: float,
    post_rank_3_multiplier: float = 1.0,
) -> float:
    if rank < 1:
        raise ValueError("rank must be 1-indexed")
    if rank == 1:
        penalty = 0
    elif rank == 2:
        penalty = start_penalty
    elif rank == 3:
        penalty = start_penalty + increment
    else:
        penalty = (
            start_penalty
            + increment
            + (increment * post_rank_3_multiplier) * (rank - 3)
        )
    base = 100 - penalty
    return base ** (rank - 1) / 100 ** (rank - 1)


def make_decay_formula(
    start_penalty: float, increment: float, post_rank_3_multiplier: float = 1.0
) -> Callable[[int], float]:
    return lambda rank: decay_multiplier(
        rank,
        start_penalty,
        increment,
        post_rank_3_multiplier=post_rank_3_multiplier,
    )


def apply_decay_to_vector(
    vector: dict[str, float],
    decay_fn: Callable[[int], float],
    exclude_emotion: bool = True,
) -> dict[str, float]:
    emotion_gate = vector.get("emotion")
    # The gate is metadata, not a ranked emotion score, so leave it untouched.
    emotion_labels = {
        k: v for k, v in vector.items() if not (exclude_emotion and k == "emotion")
    }
    sorted_emotions = sorted(
        emotion_labels.items(), key=lambda item: item[1], reverse=True
    )

    decayed = {}
    for rank, (emotion_name, score) in enumerate(sorted_emotions, start=1):
        decayed[emotion_name] = score * decay_fn(rank)

    if emotion_gate is not None:
        decayed["emotion"] = emotion_gate
    return decayed


def apply_decay_to_dataset(
    vectors: list[dict], decay_fn: Callable[[int], float]
) -> list[dict]:
    return [apply_decay_to_vector(vector, decay_fn) for vector in vectors]


def apply_threshold_penalty_to_vector(
    vector: dict[str, float],
    threshold: float,
    penalty_multiplier: float,
    *,
    exclude_emotion: bool = True,
) -> dict[str, float]:
    """Apply a global penalty when the top emotion score is below threshold."""
    emotion_gate = vector.get("emotion")
    emotion_labels = {
        k: v for k, v in vector.items() if not (exclude_emotion and k == "emotion")
    }
    if not emotion_labels:
        return dict(vector)

    top_score = max(emotion_labels.values())
    if top_score >= threshold:
        penalized = dict(emotion_labels)
    else:
        penalized = {
            emotion_name: score * penalty_multiplier
            for emotion_name, score in emotion_labels.items()
        }

    if emotion_gate is not None:
        penalized["emotion"] = emotion_gate
    return penalized


def apply_gate_below_threshold_penalty_to_vector(
    vector: dict[str, float],
    top_score_threshold: float,
    score_threshold: float,
    penalty_multiplier: float,
    *,
    exclude_emotion: bool = True,
) -> dict[str, float]:
    """Penalize only scores below a threshold when the top score is high enough."""
    emotion_gate = vector.get("emotion")
    emotion_labels = {
        k: v for k, v in vector.items() if not (exclude_emotion and k == "emotion")
    }
    if not emotion_labels:
        return dict(vector)

    top_score = max(emotion_labels.values())
    if top_score <= top_score_threshold:
        penalized = dict(emotion_labels)
    else:
        penalized = {
            emotion_name: (
                score * penalty_multiplier if score < score_threshold else score
            )
            for emotion_name, score in emotion_labels.items()
        }

    if emotion_gate is not None:
        penalized["emotion"] = emotion_gate
    return penalized


def apply_threshold_penalty_to_dataset(
    vectors: list[dict], threshold: float, penalty_multiplier: float
) -> list[dict]:
    return [
        apply_threshold_penalty_to_vector(vector, threshold, penalty_multiplier)
        for vector in vectors
    ]


def apply_gate_below_threshold_penalty_to_dataset(
    vectors: list[dict],
    top_score_threshold: float,
    score_threshold: float,
    penalty_multiplier: float,
) -> list[dict]:
    return [
        apply_gate_below_threshold_penalty_to_vector(
            vector,
            top_score_threshold,
            score_threshold,
            penalty_multiplier,
        )
        for vector in vectors
    ]


def apply_decay_and_threshold_penalty_to_vector(
    vector: dict[str, float],
    decay_fn: Callable[[int], float],
    threshold: float | None = None,
    penalty_multiplier: float = 1.0,
) -> dict[str, float]:
    decayed = apply_decay_to_vector(vector, decay_fn)
    if threshold is None:
        return decayed
    return apply_threshold_penalty_to_vector(decayed, threshold, penalty_multiplier)


def apply_decay_and_threshold_penalty_to_dataset(
    vectors: list[dict],
    decay_fn: Callable[[int], float],
    threshold: float | None = None,
    penalty_multiplier: float = 1.0,
) -> list[dict]:
    return [
        apply_decay_and_threshold_penalty_to_vector(
            vector, decay_fn, threshold=threshold, penalty_multiplier=penalty_multiplier
        )
        for vector in vectors
    ]


def apply_decay_and_gate_below_threshold_penalty_to_dataset(
    vectors: list[dict],
    decay_fn: Callable[[int], float],
    top_score_threshold: float,
    score_threshold: float,
    penalty_multiplier: float,
) -> list[dict]:
    return [
        apply_gate_below_threshold_penalty_to_vector(
            apply_decay_to_vector(vector, decay_fn),
            top_score_threshold,
            score_threshold,
            penalty_multiplier,
        )
        for vector in vectors
    ]


def compare_vectors_side_by_side(original: dict, decayed: dict, top_n: int = 5) -> str:
    original_emotions = {k: v for k, v in original.items() if k != "emotion"}
    decayed_emotions = {k: v for k, v in decayed.items() if k != "emotion"}
    sorted_original = sorted(
        original_emotions.items(), key=lambda item: item[1], reverse=True
    )[:top_n]

    lines = ["Rank | Emotion    | Original | Decayed  | Multiplier", "-" * 55]
    for rank, (emotion, orig_score) in enumerate(sorted_original, start=1):
        decayed_score = decayed_emotions[emotion]
        multiplier = decayed_score / orig_score if orig_score > 0 else 0
        lines.append(
            f"{rank:4d} | {emotion:10s} | {orig_score:.4f}   | {decayed_score:.4f}   | {multiplier:.4f}"
        )
    return "\n".join(lines)


def compute_stats_for_vectors(
    vectors: list[dict], exclude_emotion: bool = True
) -> dict[str, dict]:
    if not vectors:
        return {}
    emotion_keys = list(vectors[0].keys())
    if exclude_emotion and "emotion" in emotion_keys:
        emotion_keys.remove("emotion")
    return {
        emotion: {
            "mean": float(np.mean([v[emotion] for v in vectors if emotion in v])),
            "std": float(np.std([v[emotion] for v in vectors if emotion in v])),
            "median": float(np.median([v[emotion] for v in vectors if emotion in v])),
            "min": float(np.min([v[emotion] for v in vectors if emotion in v])),
            "max": float(np.max([v[emotion] for v in vectors if emotion in v])),
        }
        for emotion in emotion_keys
    }


def compute_rank_stats(
    vectors: list[dict], exclude_emotion: bool = True
) -> dict[int, dict]:
    rank_scores: dict[int, list[float]] = defaultdict(list)
    for vector in vectors:
        emotions = {
            k: v for k, v in vector.items() if not (exclude_emotion and k == "emotion")
        }
        for rank, (_, score) in enumerate(
            sorted(emotions.items(), key=lambda item: item[1], reverse=True), start=1
        ):
            rank_scores[rank].append(score)

    return {
        rank: {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "count": len(scores),
        }
        for rank, scores in sorted(rank_scores.items())
    }


# %%
def analyze_emotion_dataframe(df: pd.DataFrame, dataset_name: str):
    vectors = extract_emotions(df)

    stats = compute_emotion_stats(vectors, dataset_name=dataset_name)
    print_emotion_stats(f"{dataset_name} - Emotion Statistics", stats)

    rank_averages = compute_top_rank_averages(vectors)
    print_rank_stats(f"{dataset_name} - Top Rank Averages", rank_averages)

    count_distribution = compute_emotion_count_distribution(vectors)
    print_count_distribution(
        f"{dataset_name} - Emotion Count Distribution", count_distribution
    )


def analyze_decayed_emotion_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    *,
    start_penalty: float = 10.0,
    increment: float = 5.0,
    post_rank_3_multiplier: float = 1.0,
    threshold: float | None = None,
    penalty_multiplier: float = 1.0,
    top_score_threshold: float | None = None,
    score_threshold: float | None = None,
) -> tuple[dict, dict, dict]:
    """Apply the decay formula to the dataset and print the decayed summary."""
    # Print out what values are used
    print("\n\n" + "=" * 80)
    print("Decay formula:")
    print("-" * 80)
    print("Start penalty:", start_penalty)
    print("Increment:", increment)
    print("Post-rank 3 multiplier:", post_rank_3_multiplier)
    print("Threshold:", threshold)
    print("Penalty multiplier:", penalty_multiplier)
    print("Top score threshold:", top_score_threshold)
    print("Score threshold:", score_threshold)
    print("-" * 80 + "\n")
    vectors = extract_emotions(df)
    decay_fn = make_decay_formula(
        start_penalty=start_penalty,
        increment=increment,
        post_rank_3_multiplier=post_rank_3_multiplier,
    )
    if top_score_threshold is not None and score_threshold is not None:
        decayed_vectors = apply_decay_and_gate_below_threshold_penalty_to_dataset(
            vectors,
            decay_fn,
            top_score_threshold=top_score_threshold,
            score_threshold=score_threshold,
            penalty_multiplier=penalty_multiplier,
        )
    else:
        decayed_vectors = apply_decay_and_threshold_penalty_to_dataset(
            vectors,
            decay_fn,
            threshold=threshold,
            penalty_multiplier=penalty_multiplier,
        )

    print("\nSample of the decay effect (first row):")
    print(compare_vectors_side_by_side(vectors[0], decayed_vectors[0], top_n=5))

    stats = compute_emotion_stats(
        decayed_vectors, dataset_name=f"{dataset_name} (decayed)"
    )
    print_emotion_stats(f"{dataset_name} - Decayed Emotion Statistics", stats)

    rank_averages = compute_top_rank_averages(decayed_vectors)
    print_rank_stats(f"{dataset_name} - Decayed Top Rank Averages", rank_averages)

    count_distribution = compute_emotion_count_distribution(decayed_vectors)
    print_count_distribution(
        f"{dataset_name} - Decayed Emotion Count Distribution",
        count_distribution,
    )
    return stats, rank_averages, count_distribution


def plot_emotion_correlation_heatmap(df):
    """
    Generates and displays a correlation heatmap of emotion scores from multiple datasets.

    Args:
        emotions_df (pd.DataFrame): DataFrame containing emotion data.
        urgency_df (pd.DataFrame): DataFrame containing urgency-related emotion data.
        arxiv_df (pd.DataFrame): DataFrame containing arXiv-related emotion data.
    """
    # Extract emotions from the dataframe
    emotion_vectors_for_corr = extract_emotions(df)

    # Convert list of dictionaries to DataFrame
    # Filter out 'emotion' if it's metadata and not a score
    emotion_scores_df = pd.DataFrame(
        [
            {k: v for k, v in vector.items() if k != "emotion"}
            for vector in emotion_vectors_for_corr
        ]
    )

    # Compute the correlation matrix
    correlation_matrix = emotion_scores_df.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title("Correlation Matrix of Emotion Scores (Emotions Dataset)")
    plt.show()

# %%
emotions_df = load_parquet_frame(LOCAL_EMOTIONS_PARQUET, EMOTIONS_PARQUET_URL)
emotions_df.head()
# %%
urgency_df = load_parquet_frame(LOCAL_URGENCY_PARQUET, URGENCY_PARQUET_URL)
urgency_df.head()
# %%
arxiv_df = load_parquet_frame(LOCAL_ARXIV_PARQUET, ARXIV_PARQUET_URL)
arxiv_df.head()
# %%%
all_df = pd.concat([emotions_df, urgency_df, arxiv_df])

# %%
# Toggle if using the machine translated multilingual one
augmented_df = load_parquet_frame(LOCAL_AUGMENTED_PARQUET, AUGMENTED_PARQUET_URL)
full_df = augmented_df if True else all_df

# %%
# Baseline analysis (original scores)
analyze_emotion_dataframe(emotions_df, "Emotions Dataset")
# %%
analyze_emotion_dataframe(urgency_df, "Urgency Dataset")
# %%
analyze_emotion_dataframe(arxiv_df, "Arxiv Dataset")
# %%
# Decay-aware analysis (use this to reduce entailment noise)
DECAY_CONFIGS = [
    (5, 2.5, 2),
    (10.0, 5.0, 1.25),
    (8.0, 4.0, 1.25),
    (12.0, 6.0, 1.30),
]

THRESHOLD_DECAY_CONFIGS = [
    (8.0, 4.0, 1.10, 0.65, 0.70),
    (10.0, 5.0, 1.25, 0.65, 0.70),
    (8.0, 4.0, 1.10, 0.65, 0.75),
    (10.0, 5.0, 1.25, 0.65, 0.75),
]

GATED_THRESHOLD_DECAY_CONFIGS = [
    (8.0, 4.0, 1.10, 0.70, 0.65, 0.85),
    (10.0, 5.0, 1.25, 0.70, 0.65, 0.85),
    (8.0, 4.0, 1.10, 0.75, 0.65, 0.85),
    (10.0, 5.0, 1.25, 0.75, 0.65, 0.85),
]

# %%

all_stats = {}
all_rank_averages = {}
all_count_distributions = {}

for i, cfg in enumerate(DECAY_CONFIGS):
    start_penalty, increment, post_rank_3_multiplier = cfg
    print(f"\n--- Analyzing Decay Config {i+1}: {cfg} ---")
    stats, rank_averages, count_distribution = analyze_decayed_emotion_dataframe(
        all_df,
        "All Datasets",
        start_penalty=start_penalty,
        increment=increment,
        post_rank_3_multiplier=post_rank_3_multiplier,
    )
    all_stats[f"Config {i+1}"] = stats
    all_rank_averages[f"Config {i+1}"] = rank_averages
    all_count_distributions[f"Config {i+1}"] = count_distribution

for i, cfg in enumerate(THRESHOLD_DECAY_CONFIGS):
    start_penalty, increment, post_rank_3_multiplier, threshold, penalty_multiplier = (
        cfg
    )
    print(f"\n--- Analyzing Threshold Decay Config {i+1}: {cfg} ---")
    stats, rank_averages, count_distribution = analyze_decayed_emotion_dataframe(
        all_df,
        "All Datasets",
        start_penalty=start_penalty,
        increment=increment,
        post_rank_3_multiplier=post_rank_3_multiplier,
        threshold=threshold,
        penalty_multiplier=penalty_multiplier,
    )
    all_stats[f"Threshold Config {i+1}"] = stats
    all_rank_averages[f"Threshold Config {i+1}"] = rank_averages
    all_count_distributions[f"Threshold Config {i+1}"] = count_distribution

for i, cfg in enumerate(GATED_THRESHOLD_DECAY_CONFIGS):
    (
        start_penalty,
        increment,
        post_rank_3_multiplier,
        top_score_threshold,
        score_threshold,
        penalty_multiplier,
    ) = cfg
    print(f"\n--- Analyzing Gated Threshold Decay Config {i+1}: {cfg} ---")
    stats, rank_averages, count_distribution = analyze_decayed_emotion_dataframe(
        all_df,
        "All Datasets",
        start_penalty=start_penalty,
        increment=increment,
        post_rank_3_multiplier=post_rank_3_multiplier,
        top_score_threshold=top_score_threshold,
        score_threshold=score_threshold,
        penalty_multiplier=penalty_multiplier,
    )
    all_stats[f"Gated Threshold Config {i+1}"] = stats
    all_rank_averages[f"Gated Threshold Config {i+1}"] = rank_averages
    all_count_distributions[f"Gated Threshold Config {i+1}"] = count_distribution

# %%
# Compare Mean Emotion Scores Across Configurations
print("\n" + "=" * 80)
print("Comparison of Mean Emotion Scores (Decayed)")
print("=" * 80)

mean_emotion_scores = {}
for config_name, stats in all_stats.items():
    mean_emotion_scores[config_name] = {
        emotion: s["mean"] for emotion, s in stats.items() if emotion != "emotion"
    }

mean_emotion_df = pd.DataFrame(mean_emotion_scores).T
mean_emotion_df.style.background_gradient(cmap="viridis", axis=0)

# %%
# Compare Mean Top Rank Averages Across Configurations
print("\n" + "=" * 80)
print("Comparison of Mean Top Rank Averages (Decayed)")
print("=" * 80)

mean_rank_averages = {}
for config_name, rank_avg in all_rank_averages.items():
    mean_rank_averages[config_name] = {rank: r["mean"] for rank, r in rank_avg.items()}

mean_rank_df = pd.DataFrame(mean_rank_averages).T
mean_rank_df.style.background_gradient(cmap="plasma", axis=0)

# %%
# Define the decay configuration identified as promising
chosen_decay_config = (8.0, 4.0, 1.25, 0.85, 0.65, 0.75)
(
    start_penalty,
    increment,
    post_rank_3_multiplier,
    top_score_threshold,
    score_threshold,
    penalty_multiplier,
) = chosen_decay_config

# Create the decay function
chosen_decay_fn = make_decay_formula(
    start_penalty=start_penalty,
    increment=increment,
    post_rank_3_multiplier=post_rank_3_multiplier,
)
# Extract original emotion vectors from all_df
original_vectors = extract_emotions(full_df)

# Apply the chosen decay function to the original vectors
decayed_vectors_final = apply_decay_and_gate_below_threshold_penalty_to_dataset(
    original_vectors,
    chosen_decay_fn,
    top_score_threshold=top_score_threshold,
    score_threshold=score_threshold,
    penalty_multiplier=penalty_multiplier,
)

# Create the decayed_df
decayed_df = pd.DataFrame(
    {"text": full_df["text"].tolist(), "emotion_vector": decayed_vectors_final}
)

print(
    "Successfully created decayed_df using configuration: "
    f"decay={chosen_decay_config}, top_score_threshold={top_score_threshold}, "
    f"score_threshold={score_threshold}, "
    f"penalty_multiplier={penalty_multiplier}"
)
decayed_df.head()
# %%
# Call the function with the existing DataFrames
plot_emotion_correlation_heatmap(all_df)
plot_emotion_correlation_heatmap(decayed_df)
# %%
decayed_df.to_parquet("emotions_decayed_multilingual_8-4-1.25-85-65-75.parquet")

# %%
full_df[["text", "emotion_vector"]].to_parquet("emotions_multilingual.parquet")

# %%
