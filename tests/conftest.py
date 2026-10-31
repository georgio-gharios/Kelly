import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def random_returns():
    """Synthetic daily returns: 3 markets, ~1% daily vol, no edge."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2000-01-03", periods=2520)
    return pd.DataFrame(rng.normal(0, 0.01, (len(idx), 3)), index=idx, columns=["A", "B", "C"])
