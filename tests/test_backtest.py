import numpy as np
import pandas as pd
import pytest

from kelly.quant import metrics, signals
from kelly.quant.backtest import backtest


def test_peeking_signal_is_neutralized_by_lag(random_returns):
    """A signal that 'knows' today's return would be a money machine if the
    backtester let you trade on it. With lag=1 it must look like noise."""
    cheat = np.sign(random_returns)
    res = backtest(cheat, random_returns, cost_bps=0)
    assert abs(metrics.sharpe(res.returns)) < 1.0


def test_lag_zero_is_refused(random_returns):
    with pytest.raises(ValueError):
        backtest(random_returns, random_returns, lag=0)


def test_buy_and_hold_matches_asset(random_returns):
    pos = pd.DataFrame(0.0, index=random_returns.index, columns=random_returns.columns)
    pos["A"] = 1.0
    res = backtest(pos, random_returns, cost_bps=0)
    pd.testing.assert_series_equal(res.returns.iloc[1:], random_returns["A"].iloc[1:], check_names=False)


def test_costs_charged_on_turnover(random_returns):
    flip = pd.DataFrame(np.where(np.arange(len(random_returns)) % 2, 1.0, -1.0)[:, None]
                        .repeat(3, 1), index=random_returns.index, columns=random_returns.columns)
    free = backtest(flip, random_returns, cost_bps=0)
    paid = backtest(flip, random_returns, cost_bps=10)
    # Flipping +1 <-> -1 trades 2x notional per market per day, 3 markets.
    expected = 2 * 3 * 10 / 1e4
    assert paid.costs.iloc[2:].mean() == pytest.approx(expected)
    assert (free.returns - paid.returns).sum() > 0


def test_tsmom_finds_planted_trend():
    """Plant persistent drift; trend following should earn a positive Sharpe."""
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2000-01-03", periods=2520)
    regime = np.repeat(rng.choice([-1, 1], 10), 252)
    r = pd.DataFrame({"T": regime * 0.002 + rng.normal(0, 0.01, len(idx))}, index=idx)
    pos = signals.vol_target(signals.tsmom(r, 63), r)
    assert metrics.sharpe(backtest(pos, r, cost_bps=1).returns) > 0.5


def test_vol_target_hits_target(random_returns):
    sig = pd.DataFrame(1.0, index=random_returns.index, columns=["A"])
    pos = signals.vol_target(sig, random_returns[["A"]], target=0.10)
    res = backtest(pos, random_returns[["A"]], cost_bps=0)
    assert metrics.ann_vol(res.returns.iloc[100:]) == pytest.approx(0.10, rel=0.15)
