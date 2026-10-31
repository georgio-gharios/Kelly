"""Downloaders for free data sources. Each returns a tidy OHLCV DataFrame
indexed by date with columns: open, high, low, close, volume.

Kept deliberately thin: no caching, no cleaning. That lives in store.py
and rolls.py so each piece can be tested on its own.
"""
from __future__ import annotations

import io
import urllib.request

import pandas as pd

from .universe import UNIVERSE

COLUMNS = ["open", "high", "low", "close", "volume"]


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df[[c for c in COLUMNS if c in df.columns]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["close"])


def fetch_yahoo(root: str, start: str = "1990-01-01") -> pd.DataFrame:
    import yfinance as yf

    ticker = UNIVERSE[root].yahoo
    df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return _tidy(df)


def fetch_stooq(root: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={UNIVERSE[root].stooq}&i=d"
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode()
    if not text.startswith("Date"):
        raise RuntimeError(f"Stooq returned no data for {root}: {text[:80]!r}")
    df = pd.read_csv(io.StringIO(text), index_col="Date", parse_dates=True)
    return _tidy(df)


def fetch_fred(series_id: str) -> pd.Series:
    """A FRED series, e.g. 'DGS3MO' (3-month T-bill) for the risk-free rate."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url, timeout=30) as r:
        df = pd.read_csv(r, index_col=0, parse_dates=True, na_values=".")
    return df.iloc[:, 0].rename(series_id).astype(float)


SOURCES = {"yahoo": fetch_yahoo, "stooq": fetch_stooq}
