"""Performance metrics on daily return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS = 252


def ann_return(r: pd.Series) -> float:
    r = r.dropna()
    years = len(r) / DAYS
    return float((1 + r).prod() ** (1 / years) - 1) if years > 0 else np.nan


def ann_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(DAYS))


def sharpe(r: pd.Series) -> float:
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(DAYS)) if sd > 0 else np.nan


def sortino(r: pd.Series) -> float:
    down = np.sqrt((r.clip(upper=0) ** 2).mean())
    return float(r.mean() / down * np.sqrt(DAYS)) if down > 0 else np.nan


def drawdown(r: pd.Series) -> pd.Series:
    eq = (1 + r.fillna(0)).cumprod()
    return eq / eq.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown(r).min())


def sharpe_tstat(r: pd.Series) -> float:
    """t-stat of mean daily return; ~ Sharpe * sqrt(years)."""
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(len(r))) if r.std() > 0 else np.nan


def summary(r: pd.Series) -> dict:
    r = r.dropna()
    mdd = max_drawdown(r)
    cagr = ann_return(r)
    return {
        "start": r.index[0].date().isoformat() if len(r) else None,
        "end": r.index[-1].date().isoformat() if len(r) else None,
        "years": round(len(r) / DAYS, 1),
        "cagr": cagr,
        "vol": ann_vol(r),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "max_dd": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "hit_rate": float((r > 0).sum() / (r != 0).sum()) if (r != 0).any() else np.nan,
        "skew": float(r.skew()),
        "t_stat": sharpe_tstat(r),
    }


def by_year(r: pd.Series) -> pd.Series:
    return r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
