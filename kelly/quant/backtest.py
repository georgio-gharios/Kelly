"""A small, transparent, vectorized backtester.

Conventions (read these; they are where backtests usually lie):
  * `positions` are target weights decided at the CLOSE of day t, as a
    fraction of capital in notional exposure (1.0 = 100% notional long).
  * They earn the return from close t to close t+1. We enforce that with
    `lag=1`; there is no way to trade on a close you've just seen.
  * Costs are charged on traded notional: |change in held weight| * cost_bps.
  * Futures are margined, so idle capital earns the risk-free rate in real
    life. Returns here are EXCESS returns; ignoring cash is standard.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import metrics as m


@dataclass
class BacktestResult:
    returns: pd.Series        # net daily portfolio returns
    gross: pd.Series          # before costs
    costs: pd.Series
    held: pd.DataFrame        # weights actually held each day
    turnover: pd.Series       # traded notional / capital per day
    contrib: pd.DataFrame     # per-market gross P&L

    def stats(self) -> dict:
        s = m.summary(self.returns)
        s["turnover_annual"] = float(self.turnover.mean() * m.DAYS)
        s["cost_drag_annual"] = float(self.costs.mean() * m.DAYS)
        return s

    def equity(self) -> pd.Series:
        return (1 + self.returns).cumprod()


def backtest(positions: pd.DataFrame, returns: pd.DataFrame,
             cost_bps: float | dict[str, float] = 1.0, lag: int = 1) -> BacktestResult:
    if lag < 1:
        raise ValueError("lag must be >= 1, otherwise the backtest trades on information it can't have")
    cols = positions.columns.intersection(returns.columns)
    idx = positions.index.intersection(returns.index)
    pos = positions.loc[idx, cols].fillna(0.0)
    ret = returns.loc[idx, cols].fillna(0.0)

    held = pos.shift(lag).fillna(0.0)
    contrib = held * ret
    gross = contrib.sum(axis=1)

    traded = held.diff().abs()
    traded.iloc[0] = held.iloc[0].abs()
    bps = pd.Series(cost_bps, index=cols) if isinstance(cost_bps, dict) else cost_bps
    costs = (traded * bps / 1e4).sum(axis=1)

    return BacktestResult(
        returns=(gross - costs).rename("net"),
        gross=gross.rename("gross"),
        costs=costs.rename("costs"),
        held=held,
        turnover=traded.sum(axis=1).rename("turnover"),
        contrib=contrib,
    )
