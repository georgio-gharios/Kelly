"""Handling futures rolls: the #1 data trap in futures research.

A futures contract expires, so a "continuous" price series is stitched from
successive contracts. Yahoo's `ES=F` is simply "whatever the front month is
today". On a roll day the series jumps from the old contract's price to the
new one's, and that jump is NOT a return anyone earned. (For ES the gap is
small; for crude oil, natgas, or hogs in contango it can be several percent.)

Two standard fixes, given the true gap at each roll:
  * difference (Panama) adjustment: shift all history before a roll by the gap
  * ratio adjustment: scale all history before a roll by new/old

Without contract-level data we don't know the gaps exactly, so we also offer a
heuristic: flag one-day moves that are extreme relative to recent volatility.
That catches roll gaps but ALSO real crashes, so flagged dates must be
inspected, not blindly deleted. That trade-off is the point of this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(prices: pd.Series | pd.DataFrame):
    """Daily returns. Non-positive prices (e.g. WTI on 2020-04-20) make
    percentage returns meaningless, so returns touching them become NaN."""
    p = prices.where(prices > 0)
    return p.pct_change(fill_method=None)


def flag_jumps(prices: pd.Series, z: float = 6.0, window: int = 60) -> pd.Series:
    """Boolean Series: True where the one-day move is a robust z-score outlier.

    Uses median absolute deviation of the *previous* `window` days, so a
    jump never inflates the yardstick it is measured against.
    """
    r = simple_returns(prices)
    med = r.rolling(window, min_periods=window // 2).median().shift(1)
    mad = (r - med).abs().rolling(window, min_periods=window // 2).median().shift(1)
    score = (r - med).abs() / (1.4826 * mad)
    return (score > z).fillna(False)


def clean_returns(prices: pd.Series | pd.DataFrame, z: float = 6.0,
                  window: int = 60) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    """Returns with flagged jumps set to 0, plus the flag mask.

    Zeroing is conservative: if the flag was a real crash, we lose that
    day's P&L for every strategy alike, which is usually better than
    crediting strategies with fake roll gains.
    """
    if isinstance(prices, pd.DataFrame):
        flags = prices.apply(lambda s: flag_jumps(s, z, window))
    else:
        flags = flag_jumps(prices, z, window)
    r = simple_returns(prices)
    return r.mask(flags, 0.0), flags


def back_adjust(prices: pd.Series, gaps: pd.Series, method: str = "difference") -> pd.Series:
    """Remove known roll gaps from a stitched series.

    `gaps` is indexed by roll date. For "difference", a gap is
    new_contract_price - old_contract_price on that date; for "ratio",
    it is new/old. History *before* each roll date is adjusted so the
    most recent prices stay real, tradeable levels.
    """
    gaps = gaps.reindex(prices.index).dropna()
    adj = prices.astype(float).copy()
    for date, g in gaps.sort_index(ascending=False).items():
        before = adj.index < date
        if method == "difference":
            adj[before] += g
        elif method == "ratio":
            adj[before] *= g
        else:
            raise ValueError(method)
    return adj


def roll_report(prices: pd.DataFrame, z: float = 6.0) -> pd.DataFrame:
    """One row per market: how many jumps were flagged and the biggest one."""
    rows = []
    for col in prices:
        s = prices[col].dropna()
        f = flag_jumps(s, z)
        r = simple_returns(s)
        biggest = r[f].abs().idxmax() if f.any() else None
        rows.append({
            "market": col,
            "days": len(s),
            "flagged": int(f.sum()),
            "per_year": round(f.sum() / max(len(s) / 252, 1e-9), 2),
            "biggest_date": biggest,
            "biggest_move": float(r[biggest]) if biggest is not None else np.nan,
        })
    return pd.DataFrame(rows).set_index("market")
