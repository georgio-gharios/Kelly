# Kelly

An LLM research agent for futures markets. It's a learning project for understanding how agents work, built from the ground up.

> Named after **John L. Kelly Jr.**, Claude Shannon's colleague at Bell Labs. His 1956 paper
> *A New Interpretation of Information Rate* turned Shannon's information theory into the
> Kelly criterion for sizing bets. Claude is named after Shannon, and this agent is named after Kelly.
>
> Later components follow the same Bell Labs / information-theory lineage:
> the critic sub-agent is **Thorp** (Ed Thorp, who used Kelly's math to beat blackjack and markets,
> and who was famously skeptical of backtests).

## Roadmap

| Phase | Build | Learn | Status |
|---|---|---|---|
| 0 | Data cache, roll handling, backtester, metrics | The domain (plain Python, no LLM) | **in progress** |
| 1 | One LLM call with `get_prices` / `run_backtest` tools | Tool use | |
| 2 | Agent loop + `run_python` sandbox | Loops, stop conditions, budgets | |
| 3 | Terminal chat → web chat | Streaming, conversation state | |
| 4 | Experiment log (SQLite) + summarization | Context & memory | |
| 5 | **Thorp**, the critic sub-agent | Multi-agent, adversarial review | |
| 6 | Benchmark questions with known answers | Evals | |

## Layout

```
kelly/data/     universe.py (32 markets), sources.py (Yahoo/Stooq/FRED), store.py (parquet cache), rolls.py
kelly/quant/    backtest.py, metrics.py, signals.py
kelly/agent/    (phase 1+)
scripts/        fetch_data.py, inspect_rolls.py, baseline_tsmom.py
tests/          synthetic-data tests for every trap we know about
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest
python scripts/fetch_data.py          # download ~32 futures markets
python scripts/inspect_rolls.py       # see the roll gaps before trusting data
python scripts/baseline_tsmom.py      # trend-following baseline
```

## Backtest conventions

- Positions are weights (notional / capital) decided at the close of day *t* and held over *t → t+1*. `lag=0` is refused.
- Costs are in bps of traded notional.
- Returns are excess returns: futures don't tie up cash, so collateral interest is ignored.
- Yahoo's `XX=F` series are stitched front-months, so roll days contain fake jumps.
  `rolls.clean_returns` zeroes robust-z outliers. That also hides some real crashes, so always check `inspect_rolls.py` first.
