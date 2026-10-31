import numpy as np
import pandas as pd
import pytest

from kelly.quant import metrics


def test_constant_return_cagr():
    r = pd.Series(0.0004, index=pd.bdate_range("2020-01-01", periods=252))
    assert metrics.ann_return(r) == pytest.approx(1.0004 ** 252 - 1)


def test_max_drawdown():
    r = pd.Series([0.10, -0.50, 0.20], index=pd.bdate_range("2020-01-01", periods=3))
    assert metrics.max_drawdown(r) == pytest.approx(-0.50)


def test_sharpe_scale():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 50_000))
    assert metrics.sharpe(r) == pytest.approx(0.05 * np.sqrt(252), rel=0.15)
