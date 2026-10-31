"""The futures universe Kelly researches.

Each market has a root symbol (CME-style), the tickers for each free data
source, a sector, and contract specs. Specs are used for contract-level P&L
later on; the Phase 0 backtester works in returns and does not need them.

NOTE: multipliers are from CME/ICE contract specs as best known; verify on
the exchange site before using them for anything that matters. Prices for
grains/softs/livestock are quoted in cents, so their multiplier is
dollars per one cent of price.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    root: str          # e.g. "ES"
    name: str
    sector: str        # equity | rates | energy | metals | ags | softs | livestock | fx
    multiplier: float  # $ per 1.0 move in the quoted price
    yahoo: str         # Yahoo Finance continuous front-month ticker

    @property
    def stooq(self) -> str:
        # Stooq convention is lowercase root + ".f". Unverified for every root.
        return f"{self.root.lower()}.f"


UNIVERSE: dict[str, Market] = {m.root: m for m in [
    # Equity indices
    Market("ES", "E-mini S&P 500", "equity", 50, "ES=F"),
    Market("NQ", "E-mini Nasdaq-100", "equity", 20, "NQ=F"),
    Market("YM", "E-mini Dow", "equity", 5, "YM=F"),
    Market("RTY", "E-mini Russell 2000", "equity", 50, "RTY=F"),
    # Rates
    Market("ZT", "2-Year T-Note", "rates", 2000, "ZT=F"),
    Market("ZF", "5-Year T-Note", "rates", 1000, "ZF=F"),
    Market("ZN", "10-Year T-Note", "rates", 1000, "ZN=F"),
    Market("ZB", "30-Year T-Bond", "rates", 1000, "ZB=F"),
    # Energy
    Market("CL", "WTI Crude Oil", "energy", 1000, "CL=F"),
    Market("BZ", "Brent Crude Oil", "energy", 1000, "BZ=F"),
    Market("NG", "Natural Gas", "energy", 10000, "NG=F"),
    Market("RB", "RBOB Gasoline", "energy", 42000, "RB=F"),
    Market("HO", "Heating Oil", "energy", 42000, "HO=F"),
    # Metals
    Market("GC", "Gold", "metals", 100, "GC=F"),
    Market("SI", "Silver", "metals", 5000, "SI=F"),
    Market("HG", "Copper", "metals", 25000, "HG=F"),
    Market("PL", "Platinum", "metals", 50, "PL=F"),
    # Grains (cents/bushel)
    Market("ZC", "Corn", "ags", 50, "ZC=F"),
    Market("ZS", "Soybeans", "ags", 50, "ZS=F"),
    Market("ZW", "Chicago Wheat", "ags", 50, "ZW=F"),
    # Softs (cents/lb, cocoa $/t)
    Market("KC", "Coffee", "softs", 375, "KC=F"),
    Market("SB", "Sugar #11", "softs", 1120, "SB=F"),
    Market("CT", "Cotton", "softs", 500, "CT=F"),
    Market("CC", "Cocoa", "softs", 10, "CC=F"),
    # Livestock (cents/lb)
    Market("LE", "Live Cattle", "livestock", 400, "LE=F"),
    Market("HE", "Lean Hogs", "livestock", 400, "HE=F"),
    # FX (USD per unit of foreign currency)
    Market("6E", "Euro FX", "fx", 125_000, "6E=F"),
    Market("6J", "Japanese Yen", "fx", 12_500_000, "6J=F"),
    Market("6B", "British Pound", "fx", 62_500, "6B=F"),
    Market("6A", "Australian Dollar", "fx", 100_000, "6A=F"),
    Market("6C", "Canadian Dollar", "fx", 100_000, "6C=F"),
    Market("6S", "Swiss Franc", "fx", 125_000, "6S=F"),
]}


def markets(sector: str | None = None) -> list[Market]:
    return [m for m in UNIVERSE.values() if sector is None or m.sector == sector]


def sectors() -> list[str]:
    return sorted({m.sector for m in UNIVERSE.values()})
