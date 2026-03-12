from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            'pandas is required. Please install dependencies from Q2/code/requirements.txt'
        ) from exc
    return pd


def load_dataset(path: Path):
    pd = _require_pandas()
    if path.exists():
        return pd.read_parquet(path)

    # fallback for chunked parquet files inside /data
    parent = path.parent
    chunks = sorted(parent.glob('split_chunk_*.pq'))
    if not chunks:
        raise FileNotFoundError(f'No parquet file found at {path} or split chunks under {parent}')
    return pd.concat([pd.read_parquet(p) for p in chunks], ignore_index=True)


def chronological_split(df, time_col: str = 'trade_date', train_ratio: float = 0.7, valid_ratio: float = 0.15):
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    return {
        'train': df.iloc[:train_end].copy(),
        'valid': df.iloc[train_end:valid_end].copy(),
        'test': df.iloc[valid_end:].copy(),
    }


def detect_leakage(split_map: dict, time_col: str = 'trade_date') -> dict:
    train_max = split_map['train'][time_col].max()
    valid_min = split_map['valid'][time_col].min()
    valid_max = split_map['valid'][time_col].max()
    test_min = split_map['test'][time_col].min()
    chronological_ok = (train_max <= valid_min) and (valid_max <= test_min)
    return {
        'train_max': str(train_max),
        'valid_min': str(valid_min),
        'valid_max': str(valid_max),
        'test_min': str(test_min),
        'chronological_ok': bool(chronological_ok),
    }


def ensure_columns_exist(df, columns: Iterable[str]) -> list[str]:
    return [c for c in columns if c in df.columns]
