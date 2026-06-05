from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
import pandas as pd

CACHE_FORMAT_VERSION = 1
DEFAULT_CACHE_META_NAME = "cache_meta.json"


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_dataset_cache(
    dataset: Dataset | DatasetDict,
    cache_dir: str | Path,
    *,
    meta: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> None:
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing cache dir: {cache_dir}")
        shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(dataset, DatasetDict):
        for split_name, split in dataset.items():
            split.to_parquet(cache_dir / f"{split_name}.parquet")
    else:
        dataset.to_parquet(cache_dir / "dataset.parquet")

    if meta is not None:
        write_json_atomic(cache_dir / DEFAULT_CACHE_META_NAME, meta)


def load_dataset_cache(cache_dir: str | Path) -> Dataset | DatasetDict:
    cache_dir = Path(cache_dir)
    split_paths = sorted(cache_dir.glob("*.parquet"))
    if not split_paths:
        raise FileNotFoundError(f"No parquet cache files found in {cache_dir}")

    if len(split_paths) == 1 and split_paths[0].stem == "dataset":
        return Dataset.from_pandas(pd.read_parquet(split_paths[0]), preserve_index=False)

    if len(split_paths) == 1 and split_paths[0].stem in {"texts", "tokenized"}:
        return Dataset.from_pandas(pd.read_parquet(split_paths[0]), preserve_index=False)

    splits: dict[str, Dataset] = {}
    for split_path in split_paths:
        splits[split_path.stem] = Dataset.from_pandas(
            pd.read_parquet(split_path), preserve_index=False
        )
    return DatasetDict(splits)


def load_parquet_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing cache file: {path}")
    return Dataset.from_pandas(pd.read_parquet(path), preserve_index=False)


def zip_cache_dir(source_dir: str | Path, zip_path: str | Path) -> Path:
    source_dir = Path(source_dir)
    zip_path = Path(zip_path)
    if zip_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing zip: {zip_path}")

    seven_zip = shutil.which("7z") or shutil.which("7za")
    zip_bin = shutil.which("zip")

    if seven_zip is not None:
        subprocess.run(
            [seven_zip, "a", "-tzip", str(zip_path), "."],
            check=True,
            cwd=source_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return zip_path

    if zip_bin is not None:
        files = [str(path.relative_to(source_dir)) for path in source_dir.rglob("*") if path.is_file()]
        if not files:
            raise FileNotFoundError(f"No files found to zip in {source_dir}")
        subprocess.run(
            [zip_bin, "-r", str(zip_path), *files],
            check=True,
            cwd=source_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return zip_path

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir))
    return zip_path


def unzip_cache_dir(zip_path: str | Path, target_dir: str | Path) -> Path:
    zip_path = Path(zip_path)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    seven_zip = shutil.which("7z") or shutil.which("7za")
    unzip = shutil.which("unzip")

    if seven_zip is not None:
        subprocess.run(
            [seven_zip, "x", "-y", f"-o{target_dir}", str(zip_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return target_dir

    if unzip is not None:
        subprocess.run(
            [unzip, "-o", str(zip_path), "-d", str(target_dir)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return target_dir

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    return target_dir
