# -*- coding: utf-8 -*-
"""
Optimized Batch Processing for Emotion Classification
Uses batch inference with transformers pipelines for 5-10x speedup
"""
#%%
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import requests
from urllib.parse import urljoin
import seaborn as sns
import matplotlib.pyplot as plt # Keep this import, even if not directly used in the diff
from tqdm.auto import tqdm
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed

tqdm.pandas()

# Load data
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

CLASSIFICATION_REGISTRY_PATH = Path("classification_server_cluster.json")

emotions_df = pd.read_parquet(EMOTIONS_PARQUET_URL)
urgency_df = pd.read_parquet(URGENCY_PARQUET_URL)
arxiv_df = pd.read_parquet(ARXIV_PARQUET_URL)

# ============================================================================
# MAPPING & CONVERSION FUNCTIONS (unchanged)
# ============================================================================

TABULARISAI_TO_ROBERTA_MAP = {
    "fear": "fear",
    "anger": "anger",
    "frustration": "anger",
    "disgust": "disgust",
    "contempt": "disgust",
    "surprise": "surprise",
    "sadness": "sadness",
    "joy": "joy",
    "love": "joy",
    "gratitude": "joy",
}


def remap_tabularisai_to_dict(tabularisai_raw_output) -> dict[str, float]:
    """Converts tabularisai pipeline output to flat dictionary"""
    aligned_dict = {
        "fear": 0.0,
        "anger": 0.0,
        "disgust": 0.0,
        "surprise": 0.0,
        "sadness": 0.0,
        "joy": 0.0,
    }

    for item in tabularisai_raw_output[0]:
        label = item["label"].lower().strip()
        score = item["score"]

        if label in TABULARISAI_TO_ROBERTA_MAP:
            target_label = TABULARISAI_TO_ROBERTA_MAP[label]
            aligned_dict[target_label] = max(aligned_dict[target_label], score)

    return aligned_dict


def flatten_pipeline_output(pipeline_raw_output) -> dict[str, float]:
    """Converts RoBERTa pipeline outputs to flat dictionary"""
    return {item["label"]: item["score"] for item in pipeline_raw_output[0]}


# ============================================================================
# BATCH PROCESSING FUNCTION
# ============================================================================


def _post_json(url: str, payload: dict, timeout: float = 300.0) -> dict:
    """Helper to send JSON POST requests."""
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Server request failed for {url}: {exc}") from exc


def call_classification_server(
    server_url: str, texts: list[str]
) -> list[list[dict[str, float]]]:
    """
    Sends texts to a classification server for inference.
    Returns the raw output from the server.
    """
    payload = {"texts": texts}
    response = _post_json(urljoin(server_url, "/infer"), payload)
    raw_outputs = response.get("raw_outputs")
    if not isinstance(raw_outputs, list):
        raise ValueError(f"Server {server_url} returned invalid raw_outputs: {raw_outputs}")
    return raw_outputs


def batch_inference(
    texts: list[str],
    server_url: str,
    model_type: str, # e.g., "roberta-raw", "tabularisai"
    batch_size: int = 64,
    remap_fn: Callable = None,
    # For Tabularisai, the server returns a list of lists of dicts,
    # where each inner list is the output for one text.
    # For RoBERTa, it's a list of lists of dicts, where each inner list is the output for one text.
    # The remap_fn is applied to the *output for a single text*.
    # So, the server should return `raw_outputs` which is a list of outputs for each text.
    # If remap_fn is provided, it will be applied to each item in `raw_outputs`.
    # Example:
    # server returns: [[{"label": "joy", "score": 0.9}], [{"label": "anger", "score": 0.8}]]
    # remap_fn will be called with [{"label": "joy", "score": 0.9}] then [{"label": "anger", "score": 0.8}]
    # This means the server should return the output for each text as a list of dicts.
    # The current `flatten_pipeline_output` and `remap_tabularisai_to_dict` expect `[output]` as input.
) -> list[dict[str, float]]:
    """
    Process texts in batches for faster inference.

    Args:
        texts: List of texts to classify
        server_url: URL of the classification server
        model_type: Identifier for the model (e.g., "roberta-raw", "tabularisai")
        batch_size: Number of texts per batch (tune based on GPU memory)
        remap_fn: Optional function to remap output (e.g., for tabularisai)

    Returns:
        List of dictionaries with emotion scores
    """
    results = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Processing batches for {model_type}"):
        batch = texts[i : i + batch_size]

        # Call the classification server
        batch_raw_outputs = call_classification_server(server_url, batch)

        # Convert outputs to dictionaries
        for raw_output_for_text in batch_raw_outputs:
            if remap_fn:
                # For tabularisai which returns nested structure
                result = remap_fn([raw_output_for_text]) # remap_fn expects a list of outputs for a single text
            else:
                # For standard pipelines (RoBERTa)
                result = {item["label"]: item["score"] for item in raw_output_for_text}
            results.append(result)

    return results


# ============================================================================
# INITIALIZE PIPELINES (on GPU if available)
# ============================================================================
print("Loading classification server registry...")

def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)

try:
    classification_registry = load_json_file(CLASSIFICATION_REGISTRY_PATH)
    servers_info = classification_registry.get("servers", [])
    
    # Map model names to server URLs
    server_urls_by_model = {}
    for server_spec in servers_info:
        model_name = server_spec.get("model")
        if model_name:
            server_urls_by_model[model_name] = server_spec["url"]

    # Define model identifiers used in the script and map them to actual model names
    ROBERTA_RAW_MODEL = "emotions-entailment/roberta-raw"
    ROBERTA_DECAY_MODEL = "emotions-entailment/roberta-8-4-1.25-65-75"
    TABULARISAI_MODEL = "tabularisai/multilingual-emotion-classification"

    entailment_raw_server_url = server_urls_by_model.get(ROBERTA_RAW_MODEL)
    entailment_decay_server_url = server_urls_by_model.get(ROBERTA_DECAY_MODEL)
    tabularisai_server_url = server_urls_by_model.get(TABULARISAI_MODEL)

    if not all([entailment_raw_server_url, entailment_decay_server_url, tabularisai_server_url]):
        missing_models = [model for model, url in {ROBERTA_RAW_MODEL: entailment_raw_server_url, ROBERTA_DECAY_MODEL: entailment_decay_server_url, TABULARISAI_MODEL: tabularisai_server_url}.items() if url is None]
        raise RuntimeError(f"Not all required classification servers are running. Missing servers for models: {', '.join(missing_models)}. Please start them using server_manager.py.")

    print("Classification servers configured!")
    print(f"  Roberta Raw: {entailment_raw_server_url}")
    print(f"  Roberta Decay: {entailment_decay_server_url}")
    print(f"  Tabularisai: {tabularisai_server_url}")

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure classification servers are started and the registry file exists.")
    exit(1)

# ============================================================================
# BATCH PROCESS DATAFRAMES CONCURRENTLY
# ============================================================================

batch_size = 32  # Adjust based on your GPU memory (larger = faster but more memory)

def _process_df_with_models(df_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Processes a single DataFrame with all three models concurrently."""
    print(f"\n" + "=" * 60)
    print(f"Processing {df_name} concurrently with 3 models...")
    print("=" * 60)

    texts_to_process = df["text"].tolist()
    
    # Define tasks for each model
    tasks = [
        (entailment_raw_server_url, "roberta-raw", None, "emotion_vector_entailment_raw"),
        (entailment_decay_server_url, "roberta-decay", None, "emotion_vector_entailment_decay"),
        (tabularisai_server_url, "tabularisai", remap_tabularisai_to_dict, "emotion_vector_tabularisai"),
    ]

    results_futures = {}
    with ThreadPoolExecutor(max_workers=3 * 2) as executor:
        for server_url, model_type, remap_fn, column_name in tasks:
            future = executor.submit(
                batch_inference,
                texts_to_process,
                server_url,
                model_type,
                batch_size,
                remap_fn,
            )
            results_futures[future] = column_name

        for future in as_completed(results_futures):
            column_name = results_futures[future]
            try:
                df[column_name] = future.result()
            except Exception as exc:
                print(f"Error processing {df_name} with model for column {column_name}: {exc}")
                # Depending on requirements, you might want to re-raise or handle gracefully
                df[column_name] = [None] * len(df) # Assign None or empty list on failure
    return df

# Process each dataframe
emotions_df = _process_df_with_models("emotions_df", emotions_df)
urgency_df = _process_df_with_models("urgency_df", urgency_df)
arxiv_df = _process_df_with_models("arxiv_df", arxiv_df)

print("\nAll DataFrames processed!")
print(f"emotions_df shape: {emotions_df.shape}")
print(f"urgency_df shape: {urgency_df.shape}")
print(f"arxiv_df shape: {arxiv_df.shape}")

# %%
# Save it
emotions_df.to_parquet("emotions_df_with_models.parquet")
urgency_df.to_parquet("urgency_df_with_models.parquet")
arxiv_df.to_parquet("arxiv_df_with_models.parquet")
# %%