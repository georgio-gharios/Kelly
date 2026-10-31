"""Local Parquet cache: download once, research offline forever.

Layout: data/cache/<source>/<ROOT>.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .sources import SOURCES

CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"


def path(root: str, source: str = "yahoo") -> Path:
    return CACHE / source / f"{root}.parquet"


def save(df: pd.DataFrame, root: str, source: str = "yahoo") -> Path:
    p = path(root, source)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    return p


def load(root: str, source: str = "yahoo", refresh: bool = False) -> pd.DataFrame:
    p = path(root, source)
    if refresh or not p.exists():
        save(SOURCES[source](root), root, source)
    return pd.read_parquet(p)


def available(source: str = "yahoo") -> list[str]:
    return sorted(p.stem for p in (CACHE / source).glob("*.parquet"))


def closes(roots: list[str] | None = None, source: str = "yahoo") -> pd.DataFrame:
    """Wide frame of close prices (date x root) from the cache only."""
    roots = roots or available(source)
    return pd.DataFrame({r: load(r, source)["close"] for r in roots}).sort_index()
