"""Look at roll gaps in the cached continuous series before trusting them.

    python scripts/inspect_rolls.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from kelly.data import rolls, store  # noqa: E402

prices = store.closes()
if prices.empty:
    raise SystemExit("Cache is empty. Run scripts/fetch_data.py first.")
pd.set_option("display.width", 120)
print(rolls.roll_report(prices).sort_values("per_year", ascending=False))
