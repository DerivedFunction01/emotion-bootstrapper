# %%
from __future__ import annotations

import math
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib import error as urllib_error
from urllib import request as urllib_request

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


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

DEFAULT_OUTPUT_PATH = Path("emotion_translation_augmented.parquet")
DEFAULT_TEMP_DIR = Path("emotion_translation_augmented_tmp")
DEFAULT_SERVER_REGISTRY = Path("translation_server_cluster.json")
DEFAULT_CHECKPOINT_PATH = DEFAULT_TEMP_DIR / "checkpoint.json"
DEFAULT_BATCH_SIZE = 32
DEFAULT_FLUSH_EVERY = 2000

SOURCE_PARQUETS = {
    "emotions": EMOTIONS_PARQUET_URL,
    "urgency": URGENCY_PARQUET_URL,
    "arxiv": ARXIV_PARQUET_URL,
}

TARGET_LANGUAGES = {
    "french": "fra_Latn",
    "german": "deu_Latn",
    "spanish": "spa_Latn",
    "italian": "ita_Latn",
    "chinese": "zho_Hans",
    "japanese": "jpn_Jpan",
    "russian": "rus_Cyrl",
    "arabic": "arb_Arab",
    "hindi": "hin_Deva",
    "portuguese": "por_Latn",
    "korean": "kor_Hang",
    "english": "eng_Latn",
}

DEFAULT_SAMPLE_FRACTION = 1 / len(TARGET_LANGUAGES)

# %%
def local_path_from_url(url: str) -> Path:
    return Path(Path(urlparse(url).path).name)


def load_parquet_frame(local_path: Path, remote_url: str) -> pd.DataFrame:
    if local_path.exists():
        print(f"Loaded {local_path} from local file system.")
        return pd.read_parquet(local_path)

    print(f"Local file {local_path} not found. Downloading from {remote_url} ...")
    frame = pd.read_parquet(remote_url)
    frame.to_parquet(local_path)
    print(f"Downloaded and saved {local_path}.")
    return frame


def batch_iterable(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@dataclass(frozen=True)
class TranslationServer:
    name: str
    url: str


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=600.0) as response:
            return json.load(response)
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Translation server request failed for {url}: {exc.code} {detail}") from exc


def _load_translation_servers(registry_path: Path) -> list[TranslationServer]:
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing translation registry: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    servers = registry.get("servers", [])
    if not servers:
        raise ValueError(f"No servers found in translation registry {registry_path}")
    return [TranslationServer(name=server["name"], url=server["url"]) for server in servers]


def _worker_count_for_servers(server_count: int) -> int:
    if server_count < 1:
        raise ValueError("server_count must be at least 1")
    return server_count * 2


def _load_checkpoint(checkpoint_path: Path) -> dict[str, list[str]]:
    if not checkpoint_path.exists():
        return {}
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    completed = payload.get("completed", {})
    if not isinstance(completed, dict):
        raise ValueError(f"Invalid checkpoint format in {checkpoint_path}")
    return {
        str(source_name): [str(lang) for lang in langs]
        for source_name, langs in completed.items()
    }


def _write_checkpoint(checkpoint_path: Path, completed: dict[str, list[str]]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".tmp")
    payload = {"completed": completed}
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(checkpoint_path)


def _translate_batch_via_server(
    server: TranslationServer,
    texts: list[str],
    source_lang: str,
    target_lang: str,
) -> list[str]:
    response = _post_json(
        f"{server.url.rstrip('/')}/translate",
        {
            "texts": texts,
            "source_lang": source_lang,
            "target_lang": target_lang,
        },
    )
    translations = response.get("translations", [])
    if not isinstance(translations, list):
        raise ValueError(f"Translation server {server.name} returned an invalid response: {response}")
    return [str(text) for text in translations]

def augment_dataframe_with_translations(
    df: pd.DataFrame,
    source_name: str,
    servers: list,
    *,
    sample_fraction: float,
    batch_size: int,
    flush_every: int,
    temp_dir: Path,
    completed_languages: set[str],
    source_lang: str = "eng_Latn",
) -> list[Path]:
    if "text" not in df.columns:
        raise KeyError(f"{source_name} is missing required 'text' column")

    # 1. Calculate step size per language
    sample_size = max(1, math.ceil(len(df) * sample_fraction))
    total_rows = len(df)

    chunk_paths: list[Path] = []
    pending_rows: list[dict] = []
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total expected progress based on variable slices
    total_translations = sample_size * len(TARGET_LANGUAGES)
    server_cycle = list(servers)
    if not server_cycle:
        raise ValueError("No translation servers available")
    max_workers = _worker_count_for_servers(len(server_cycle))

    with tqdm(total=total_translations, desc=f"{source_name} translations") as progress:
        # Enumerate languages so we can track our sliding position
        for lang_idx, (target_name, target_lang) in enumerate(TARGET_LANGUAGES.items()):
            if target_name in completed_languages:
                print(
                    f"Skipping {source_name}: {target_name} already completed in checkpoint."
                )
                progress.update(sample_size)
                continue

            # 2. Dynamic Sliding Window with Wraparound
            start_idx = (lang_idx * sample_size) % total_rows
            end_idx = start_idx + sample_size

            if end_idx <= total_rows:
                # Normal slice fits within boundaries
                sampled_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
            else:
                # Wraparound slice: take till the end, then wrap to the beginning
                part1 = df.iloc[start_idx:total_rows]
                part2 = df.iloc[0 : (end_idx % total_rows)]
                sampled_df = pd.concat([part1, part2], ignore_index=True)

            print(
                f"Translating {source_name}: {target_name} ({target_lang}) | Rows {start_idx} to {(end_idx - 1) % total_rows}"
            )

            # Extract lists specific to this language's unique sample slice
            texts = sampled_df["text"].astype(str).tolist()
            row_records = sampled_df.to_dict(orient="records")

            batches = list(batch_iterable(texts, batch_size))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_meta = {}
                for batch_index, batch_texts in enumerate(
                    tqdm(
                        batches,
                        total=math.ceil(len(texts) / batch_size),
                        desc=f"{source_name}->{target_name}",
                        leave=False,
                    )
                ):
                    server = server_cycle[batch_index % len(server_cycle)]
                    future = executor.submit(
                        _translate_batch_via_server,
                        server,
                        batch_texts,
                        source_lang,
                        target_lang,
                    )
                    future_to_meta[future] = (batch_index, batch_texts)

                ordered_results: dict[int, list[str]] = {}
                next_batch_index = 0
                for future in as_completed(future_to_meta):
                    batch_index, batch_texts = future_to_meta[future]
                    ordered_results[batch_index] = future.result()

                    while next_batch_index in ordered_results:
                        translated_texts = ordered_results.pop(next_batch_index)

                        # Grab rows matching this specific batch from our current dynamic sample
                        batch_rows = row_records[
                            next_batch_index
                            * batch_size : next_batch_index
                            * batch_size
                            + len(translated_texts)
                        ]

                        for row, translated_text in zip(batch_rows, translated_texts):
                            augmented_row = dict(row)
                            augmented_row["text"] = translated_text
                            augmented_row["source_dataset"] = source_name
                            augmented_row["translation_language"] = target_name
                            augmented_row["translation_language_code"] = target_lang
                            augmented_row["translation_source_text"] = row["text"]
                            augmented_row["is_translation"] = True
                            pending_rows.append(augmented_row)

                            if len(pending_rows) >= flush_every:
                                chunk_path = temp_dir / (
                                    f"{source_name}_{target_name}_chunk_{len(chunk_paths):05d}.parquet"
                                )
                                pd.DataFrame(pending_rows).to_parquet(
                                    chunk_path, index=False
                                )
                                chunk_paths.append(chunk_path)
                                pending_rows.clear()

                        progress.update(len(translated_texts))
                        next_batch_index += 1

            # Ensure we flush remaining rows *per language* so chunks don't leak language to language
            if pending_rows:
                chunk_path = temp_dir / (
                    f"{source_name}_{target_name}_chunk_{len(chunk_paths):05d}.parquet"
                )
                pd.DataFrame(pending_rows).to_parquet(chunk_path, index=False)
                chunk_paths.append(chunk_path)
                pending_rows.clear()

            completed_languages.add(target_name)

    return chunk_paths


def write_parquet_chunks(chunk_paths: list[Path], output_path: Path) -> None:
    if not chunk_paths:
        raise ValueError("No chunk files were produced; nothing to write.")

    writer: pq.ParquetWriter | None = None
    try:
        for chunk_path in tqdm(chunk_paths, desc="Writing final parquet"):
            table = pa.Table.from_pandas(pd.read_parquet(chunk_path), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def build_augmented_dataset(
    output_path: Path,
    registry_path: Path = DEFAULT_SERVER_REGISTRY,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    flush_every: int = DEFAULT_FLUSH_EVERY,
    temp_dir: Path = DEFAULT_TEMP_DIR,
) -> pd.DataFrame:
    servers = _load_translation_servers(registry_path)
    temp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _load_checkpoint(checkpoint_path)

    chunk_paths: list[Path] = sorted(temp_dir.glob("*.parquet"))

    for source_name, remote_url in tqdm(
        list(SOURCE_PARQUETS.items()), desc="Sources"
    ):
        local_path = local_path_from_url(remote_url)
        frame = load_parquet_frame(local_path, remote_url)
        completed_languages = set(checkpoint.get(source_name, []))
        chunk_paths.extend(
            augment_dataframe_with_translations(
                frame,
                source_name,
                servers,
                sample_fraction=sample_fraction,
                batch_size=batch_size,
                flush_every=flush_every,
                temp_dir=temp_dir,
                completed_languages=completed_languages,
            )
        )
        checkpoint[source_name] = sorted(completed_languages)
        _write_checkpoint(checkpoint_path, checkpoint)

    chunk_paths = sorted(set(chunk_paths))
    write_parquet_chunks(chunk_paths, output_path)
    print(f"Saved augmented parquet to {output_path}")
    return pd.read_parquet(output_path)

# %%
build_augmented_dataset(DEFAULT_OUTPUT_PATH)
