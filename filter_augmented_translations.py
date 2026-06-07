from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DEFAULT_INPUT_PATH = Path("emotion_translation_augmented.parquet")
DEFAULT_OUTPUT_PATH = Path("emotion_translation_augmented_filtered.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter bad rows from the merged translation parquet."
    )
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _has_garbage_symbols(text: str) -> bool:
    if not text:
        return True
    if text.lower() in {"unk", "unknown", "<unk>", "[unk]"}:
        return True

    alpha_count = sum(1 for ch in text if ch.isalpha())
    if alpha_count == 0:
        return True

    # Drop rows that are only emoji/symbol fragments, including skin tone modifiers.
    meaningful_chars = [
        ch
        for ch in text
        if unicodedata.category(ch)[0] in {"L", "N"}
    ]
    if not meaningful_chars:
        return True

    # Drop rows that look like broken tokenizer output or mostly symbol soup.
    symbol_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    if len(text) > 0 and symbol_count / len(text) > 0.35:
        return True

    return False


def filter_translated_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns:
        raise KeyError("Input parquet must contain a 'text' column")

    work = df.copy()
    work["text"] = work["text"].map(_normalize_text)
    work["translation_source_text"] = work.get("translation_source_text", "").map(
        _normalize_text
    )

    before = len(work)
    work = work[work["text"].astype(str).str.len() > 0]
    work = work[~work["text"].map(_has_garbage_symbols)]
    work = work.drop_duplicates(subset=["text", "translation_language_code"], keep="first")
    after = len(work)
    print(f"Filtered {before - after:,} row(s); kept {after:,}.")
    work = work.sample(frac=1, random_state=42)
    return work.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input parquet: {input_path}")

    print(f"Loading {input_path}")
    df = pd.read_parquet(input_path)
    filtered_df = filter_translated_rows(df)
    filtered_df.to_parquet(output_path, index=False)
    print(f"Saved filtered parquet to {output_path}")


if __name__ == "__main__":
    main()
