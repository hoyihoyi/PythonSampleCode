"""
Pillar 4: S&P 500 Sector Rotation Matrix

Downloads 60 days of daily closing prices for the 11 Select Sector SPDR ETFs
and SPY as benchmark, then computes:
    • Absolute returns over 5-day, 20-day, and 60-day windows.
    • Relative excess returns vs SPY over each window.
    • An equal-weighted composite momentum score (mean of the three relative returns).
    • Momentum acceleration: 5-day relative score minus 60-day relative score
      (positive → capital is accelerating into the sector).
    • A rule-based signal label: LEADING / ROTATING_IN / NEUTRAL / ROTATING_OUT / LAGGING.
    • A broad market regime flag (RISK_ON vs RISK_OFF) based on cyclical vs
      defensive sector aggregate scores.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TypedDict

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

SECTORS: dict[str, str] = {
    "XLF":  "Financials",
    "XLK":  "Technology",
    "XLE":  "Energy",
    "XLV":  "Health Care",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLU":  "Utilities",
    "XLC":  "Communication Services",
}
BENCHMARK    = "SPY"
LOOKBACK     = 65   # fetch extra rows so the 60-day window is always full
WINDOWS      = {"5d": 5, "20d": 20, "60d": 60}

CYCLICAL_TICKERS  = {"XLK", "XLF", "XLY", "XLE", "XLI", "XLC"}
DEFENSIVE_TICKERS = {"XLU", "XLP", "XLRE", "XLV", "XLB"}


# ── TypedDict schemas ─────────────────────────────────────────────────────────

class SectorEntry(TypedDict):
    ticker: str
    sector: str
    rank: int
    composite_score_pct: float
    ret_5d_pct: float
    ret_20d_pct: float
    ret_60d_pct: float
    rel_5d_pct: float
    rel_20d_pct: float
    rel_60d_pct: float
    momentum_acceleration_pct: float  # rel_5d − rel_60d; positive = accelerating
    signal: str


class SectorResult(TypedDict):
    leaderboard: list[SectorEntry]
    top_sector: str
    bottom_sector: str
    rotation_signal: str  # RISK_ON | RISK_OFF
    spy_ret_5d_pct: float
    spy_ret_20d_pct: float
    spy_ret_60d_pct: float
    timestamp: str
    status: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct_return(prices: pd.Series, window: int) -> float:
    """Simple price return over `window` bars; NaN if insufficient data."""
    if len(prices) <= window or prices.iloc[-(window + 1)] <= 0:
        return float("nan")
    return float((prices.iloc[-1] / prices.iloc[-(window + 1)]) - 1.0)


def _classify_signal(rel_5d: float, rel_20d: float, rel_60d: float) -> str:
    """
    Deterministic 4-state classification using relative-return sign agreement
    and momentum direction.
    """
    if any(np.isnan(v) for v in (rel_5d, rel_20d, rel_60d)):
        return "NEUTRAL"
    positives = sum(v > 0 for v in (rel_5d, rel_20d, rel_60d))
    # Acceleration: is the recent window improving vs. the medium term?
    accelerating = rel_5d > rel_20d > rel_60d
    decelerating = rel_5d < rel_20d < rel_60d

    if positives == 3 and accelerating:
        return "LEADING"
    if positives >= 2:
        return "ROTATING_IN"
    if positives == 0 and decelerating:
        return "LAGGING"
    return "ROTATING_OUT"


def _normalise_close(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extract a clean (dates × tickers) Close DataFrame from yfinance output,
    handling both MultiIndex and flat column structures.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw.xs("Close", axis=1, level=0)
    elif "Close" in raw.columns:
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.squeeze(axis=1)
        if isinstance(close, pd.Series):
            close = close.to_frame()
    else:
        close = raw  # assume already a price frame

    return close.dropna(how="all")


# ── Public entry point ────────────────────────────────────────────────────────

def run() -> SectorResult:
    """Execute Pillar 4: Sector Rotation Matrix."""
    tickers = list(SECTORS.keys()) + [BENCHMARK]
    logger.info("[SECTOR] Downloading %d tickers × %d day lookback", len(tickers), LOOKBACK)

    raw = yf.download(
        tickers,
        period="4mo",
        interval="1d",
        progress=False,
        auto_adjust=True,
        group_by="column",
    )

    if raw.empty:
        raise RuntimeError("yfinance returned empty data for sector rotation")

    close = _normalise_close(raw).iloc[-LOOKBACK:]

    if BENCHMARK not in close.columns:
        raise ValueError(f"Benchmark {BENCHMARK} not in downloaded data")

    spy_prices = close[BENCHMARK]
    spy_ret: dict[str, float] = {w: _pct_return(spy_prices, n) for w, n in WINDOWS.items()}

    entries: list[SectorEntry] = []

    for ticker, sector_name in SECTORS.items():
        if ticker not in close.columns:
            logger.warning("[SECTOR] %s not in price data — skipping", ticker)
            continue

        prices = close[ticker]
        rets   = {w: _pct_return(prices, n) for w, n in WINDOWS.items()}

        def _rel(w: str) -> float:
            r, b = rets[w], spy_ret[w]
            return float(r - b) if not (np.isnan(r) or np.isnan(b)) else float("nan")

        rel = {w: _rel(w) for w in WINDOWS}

        valid_rels = [v for v in rel.values() if not np.isnan(v)]
        composite  = float(np.mean(valid_rels)) if valid_rels else float("nan")

        accel = (float(rel["5d"] - rel["60d"])
                 if not (np.isnan(rel["5d"]) or np.isnan(rel["60d"])) else float("nan"))

        entries.append({
            "ticker":                   ticker,
            "sector":                   sector_name,
            "rank":                     0,           # assigned below
            "composite_score_pct":      round(composite * 100, 4)    if not np.isnan(composite)   else float("nan"),
            "ret_5d_pct":               round(rets["5d"]  * 100, 4)  if not np.isnan(rets["5d"])  else float("nan"),
            "ret_20d_pct":              round(rets["20d"] * 100, 4)  if not np.isnan(rets["20d"]) else float("nan"),
            "ret_60d_pct":              round(rets["60d"] * 100, 4)  if not np.isnan(rets["60d"]) else float("nan"),
            "rel_5d_pct":               round(rel["5d"]  * 100, 4)   if not np.isnan(rel["5d"])   else float("nan"),
            "rel_20d_pct":              round(rel["20d"] * 100, 4)   if not np.isnan(rel["20d"])  else float("nan"),
            "rel_60d_pct":              round(rel["60d"] * 100, 4)   if not np.isnan(rel["60d"])  else float("nan"),
            "momentum_acceleration_pct": round(accel * 100, 4)       if not np.isnan(accel)       else float("nan"),
            "signal":                   _classify_signal(rel["5d"], rel["20d"], rel["60d"]),
        })

    # ── Rank by composite score (descending) ─────────────────────────────────
    entries.sort(
        key=lambda e: e["composite_score_pct"] if not np.isnan(e["composite_score_pct"]) else -9999,
        reverse=True,
    )
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank

    top_sector    = entries[0]["sector"]  if entries else "N/A"
    bottom_sector = entries[-1]["sector"] if entries else "N/A"

    # ── Cyclical vs Defensive regime ─────────────────────────────────────────
    cyc_scores = [e["composite_score_pct"] for e in entries
                  if e["ticker"] in CYCLICAL_TICKERS and not np.isnan(e["composite_score_pct"])]
    def_scores = [e["composite_score_pct"] for e in entries
                  if e["ticker"] in DEFENSIVE_TICKERS and not np.isnan(e["composite_score_pct"])]
    cyc_mean = float(np.mean(cyc_scores)) if cyc_scores else 0.0
    def_mean = float(np.mean(def_scores)) if def_scores else 0.0
    rotation_signal = "RISK_ON" if cyc_mean > def_mean else "RISK_OFF"

    logger.info(
        "[SECTOR] Regime=%s | Top=%s | Bottom=%s | Cyclical=%.2f%% | Defensive=%.2f%%",
        rotation_signal, top_sector, bottom_sector, cyc_mean, def_mean,
    )

    return {
        "leaderboard":       entries,
        "top_sector":        top_sector,
        "bottom_sector":     bottom_sector,
        "rotation_signal":   rotation_signal,
        "spy_ret_5d_pct":    round(spy_ret["5d"]  * 100, 4) if not np.isnan(spy_ret["5d"])  else float("nan"),
        "spy_ret_20d_pct":   round(spy_ret["20d"] * 100, 4) if not np.isnan(spy_ret["20d"]) else float("nan"),
        "spy_ret_60d_pct":   round(spy_ret["60d"] * 100, 4) if not np.isnan(spy_ret["60d"]) else float("nan"),
        "timestamp":         datetime.now(tz=timezone.utc).isoformat(),
        "status":            "ok",
    }
