"""Download the futures universe into the local Parquet cache.

    python scripts/fetch_data.py                 # everything, from Yahoo
    python scripts/fetch_data.py ES CL GC        # a few markets
    python scripts/fetch_data.py --source stooq  # alternate source
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kelly.data import store  # noqa: E402
from kelly.data.universe import UNIVERSE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=list(UNIVERSE))
    ap.add_argument("--source", default="yahoo", choices=["yahoo", "stooq"])
    args = ap.parse_args()

    failed = []
    for root in args.roots:
        try:
            df = store.load(root, args.source, refresh=True)
            print(f"{root:4} {len(df):6} rows  {df.index[0].date()} -> {df.index[-1].date()}")
        except Exception as e:  # keep going; report at the end
            failed.append(root)
            print(f"{root:4} FAILED: {e}")
    if failed:
        print(f"\n{len(failed)} failed: {' '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
