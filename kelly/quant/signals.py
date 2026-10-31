"""Signal building blocks. Every function maps information available at
the close of day t to a position for day t (the backtester lags it)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewm_vol(returns: pd.DataFrame, span: int = 60) -> pd.DataFrame:
    """Annualized exponentially-weighted volatility estimate."""
    return returns.ewm(span=span, min_periods=span // 2).std() * np.sqrt(252)


def tsmom(returns: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Time-series momentum: +1 if the trailing `lookback` return is positive,
    -1 if negative (Moskowitz, Ooi & Pedersen 2012)."""
    trailing = np.log1p(returns.fillna(0)).rolling(lookback, min_periods=lookback).sum()
    return np.sign(trailing)


def vol_target(signal: pd.DataFrame, returns: pd.DataFrame, target: float = 0.10,
               span: int = 60, max_leverage: float = 5.0) -> pd.DataFrame:
    """Scale each market so it contributes `target` annual vol on its own,
    then split risk equally across the markets active that day. The result
    is a portfolio of roughly `target` vol if markets were uncorrelated."""
    vol = ewm_vol(returns, span)
    per_asset = (signal * target / vol).clip(-max_leverage, max_leverage)
    active = per_asset.notna() & (per_asset != 0)
    n = active.sum(axis=1).replace(0, np.nan)
    return per_asset.div(n, axis=0).fillna(0.0)
