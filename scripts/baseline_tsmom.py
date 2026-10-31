"""Phase 0 baseline: time-series momentum across the cached universe.

This is the benchmark the agent must later reproduce on its own.

    python scripts/baseline_tsmom.py [--cost-bps 2] [--lookback 252]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kelly.data import rolls, store  # noqa: E402
from kelly.quant import metrics, signals  # noqa: E402
from kelly.quant.backtest import backtest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lookback", type=int, default=252)
ap.add_argument("--target-vol", type=float, default=0.10)
ap.add_argument("--cost-bps", type=float, default=2.0)
ap.add_argument("--out", default="reports/baseline_tsmom.png")
args = ap.parse_args()

prices = store.closes()
if prices.empty:
    raise SystemExit("Cache is empty. Run scripts/fetch_data.py first.")
rets, flags = rolls.clean_returns(prices)
print(f"{len(prices.columns)} markets, {int(flags.sum().sum())} jump days zeroed")

pos = signals.vol_target(signals.tsmom(rets, args.lookback), rets, args.target_vol)
res = backtest(pos, rets, cost_bps=args.cost_bps)
r = res.returns[res.held.abs().sum(axis=1) > 0]

stats = {**metrics.summary(r), "turnover_annual": float(res.turnover.mean() * 252)}
for k, v in stats.items():
    print(f"{k:18} {v:.3f}" if isinstance(v, float) else f"{k:18} {v}")
print("\nBy year:\n", metrics.by_year(r).round(3).to_string())

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[3, 1])
(1 + r).cumprod().plot(ax=a, title="TSMOM baseline (net of costs)", logy=True)
metrics.drawdown(r).plot(ax=b, title="Drawdown", color="firebrick")
fig.tight_layout()
fig.savefig(args.out, dpi=120)
print(f"\nChart: {args.out}")
