import numpy as np
import pandas as pd
import pytest

from kelly.data import rolls


def make_stitched():
    """A smooth 'true' price plus a +5 roll gap every 63 days, like a
    contract in steep backwardation stitched front-to-front."""
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2010-01-04", periods=756)
    true = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.005, len(idx)))), index=idx)
    roll_dates = idx[63::63]
    gaps = pd.Series(5.0, index=roll_dates)
    stitched = true + gaps.reindex(idx).fillna(0).cumsum()
    return true, stitched, gaps


def test_flag_jumps_finds_roll_gaps():
    _, stitched, gaps = make_stitched()
    flags = rolls.flag_jumps(stitched)
    found = set(flags[flags].index)
    assert set(gaps.index[1:]) <= found          # all gaps after warm-up found
    assert len(found - set(gaps.index)) <= 2     # few false alarms


def test_back_adjust_difference_restores_returns():
    true, stitched, gaps = make_stitched()
    # Gap as new - old; history before each roll shifts UP by it.
    adj = rolls.back_adjust(stitched, gaps, "difference")
    np.testing.assert_allclose(adj.diff().dropna(), true.diff().dropna(), atol=1e-9)


def test_nonpositive_prices_give_nan_returns():
    p = pd.Series([20.0, 10.0, -37.6, 10.0], index=pd.bdate_range("2020-04-16", periods=4))
    r = rolls.simple_returns(p)
    assert r.iloc[1] == pytest.approx(-0.5)
    assert r.iloc[2:].isna().all()
