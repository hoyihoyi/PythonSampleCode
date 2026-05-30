"""
Pillar 2: Systematic & CTA Flow Tracker

Replicates core Time-Series Momentum (TSMOM) rules inspired by Moskowitz,
Ooi & Pedersen (2012). For each asset we compute:
    • 50-day and 200-day Simple Moving Averages (+ Golden/Death Cross detection)
    • 20-day and 100-day Donchian Channel Highs/Lows (breakout flags)
    • 12-month annualised return, rolling daily volatility, Sharpe-12M
    • A deterministic 6-factor binary trend classification

These flags surface the structural pivot zones where CTAs and systematic funds
are mechanically forced to reverse their momentum positioning.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

ASSETS: dict[str, str] = {
    "SPY":     "S&P 500",
    "QQQ":     "Nasdaq 100",
    "GLD":     "Gold",
    "USO":     "Crude Oil",
    "BTC-USD": "Bitcoin",
}

LOOKBACK_DAYS = 252
RISK_FREE_RATE = 0.05  # annual, for Sharpe calculation


# ── TypedDict schemas ─────────────────────────────────────────────────────────

class AssetCTAState(TypedDict):
    ticker: str
    name: str
    spot: float
    sma_50: float
    sma_200: float
    is_above_50sma: bool
    is_above_200sma: bool
    golden_cross: bool   # 50 crossed above 200 today
    death_cross: bool    # 50 crossed below 200 today
    donchian_high_20: float
    donchian_low_20: float
    donchian_high_100: float
    donchian_low_100: float
    is_breakout_high_20: bool
    is_breakout_low_20: bool
    is_breakout_high_100: bool
    is_breakout_low_100: bool
    return_12m: float
    volatility_annualized: float
    sharpe_12m: float
    trend_signal: str  # STRONG_BULL | BULL | NEUTRAL | BEAR | STRONG_BEAR


class CTAResult(TypedDict):
    assets: list[Any]  # AssetCTAState or error dict
    timestamp: str
    status: str


# ── Single-asset computation ──────────────────────────────────────────────────

def _squeeze_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Extract a clean 1-D Series from a potentially MultiIndex column DataFrame."""
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.squeeze().dropna()


def _compute_asset(ticker: str, name: str) -> AssetCTAState:
    raw = yf.download(
        ticker,
        period="2y",
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    if raw.empty or len(raw) < LOOKBACK_DAYS:
        raise ValueError(f"Insufficient history for {ticker}: {len(raw)} rows")

    close = _squeeze_series(raw, "Close").iloc[-LOOKBACK_DAYS:]
    high  = _squeeze_series(raw, "High").iloc[-LOOKBACK_DAYS:]
    low   = _squeeze_series(raw, "Low").iloc[-LOOKBACK_DAYS:]

    spot = float(close.iloc[-1])

    # ── Moving averages ───────────────────────────────────────────────────────
    sma50_series  = close.rolling(50).mean()
    sma200_series = close.rolling(200).mean()
    sma_50  = float(sma50_series.iloc[-1])
    sma_200 = float(sma200_series.iloc[-1])

    # Golden / Death Cross: today's cross vs yesterday's alignment
    prev_sma50  = float(sma50_series.iloc[-2])
    prev_sma200 = float(sma200_series.iloc[-2])
    golden_cross = bool((prev_sma50 <= prev_sma200) and (sma_50 > sma_200))
    death_cross  = bool((prev_sma50 >= prev_sma200) and (sma_50 < sma_200))

    # ── Donchian Channels ─────────────────────────────────────────────────────
    don_hi_20  = float(high.rolling(20).max().iloc[-1])
    don_lo_20  = float(low.rolling(20).min().iloc[-1])
    don_hi_100 = float(high.rolling(100).max().iloc[-1])
    don_lo_100 = float(low.rolling(100).min().iloc[-1])

    # ── 12-month momentum ─────────────────────────────────────────────────────
    ret_12m = float((close.iloc[-1] / close.iloc[0]) - 1.0)

    log_ret = np.log(close / close.shift(1)).dropna()
    vol_ann = float(log_ret.std(ddof=1) * np.sqrt(252))

    sharpe = float((ret_12m - RISK_FREE_RATE) / vol_ann) if vol_ann > 0.0 else 0.0

    # ── Trend classification (6 binary signals, thresholds at 5/3) ───────────
    bull_score = sum([
        spot > sma_50,
        spot > sma_200,
        golden_cross,
        spot >= don_hi_20,
        ret_12m > 0.0,
        sharpe > 0.5,
    ])
    bear_score = sum([
        spot < sma_50,
        spot < sma_200,
        death_cross,
        spot <= don_lo_20,
        ret_12m < 0.0,
        sharpe < -0.5,
    ])

    if bull_score >= 5:
        trend = "STRONG_BULL"
    elif bull_score >= 3:
        trend = "BULL"
    elif bear_score >= 5:
        trend = "STRONG_BEAR"
    elif bear_score >= 3:
        trend = "BEAR"
    else:
        trend = "NEUTRAL"

    return {
        "ticker":                ticker,
        "name":                  name,
        "spot":                  round(spot, 4),
        "sma_50":                round(sma_50, 4),
        "sma_200":               round(sma_200, 4),
        "is_above_50sma":        bool(spot > sma_50),
        "is_above_200sma":       bool(spot > sma_200),
        "golden_cross":          golden_cross,
        "death_cross":           death_cross,
        "donchian_high_20":      round(don_hi_20, 4),
        "donchian_low_20":       round(don_lo_20, 4),
        "donchian_high_100":     round(don_hi_100, 4),
        "donchian_low_100":      round(don_lo_100, 4),
        "is_breakout_high_20":   bool(spot >= don_hi_20),
        "is_breakout_low_20":    bool(spot <= don_lo_20),
        "is_breakout_high_100":  bool(spot >= don_hi_100),
        "is_breakout_low_100":   bool(spot <= don_lo_100),
        "return_12m":            round(ret_12m, 6),
        "volatility_annualized": round(vol_ann, 6),
        "sharpe_12m":            round(sharpe, 4),
        "trend_signal":          trend,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run() -> CTAResult:
    """Execute Pillar 2: CTA / Systematic Flow Tracker."""
    logger.info("[CTA] Processing %d assets", len(ASSETS))
    results: list[Any] = []
    ok_count = 0

    for ticker, name in ASSETS.items():
        try:
            state = _compute_asset(ticker, name)
            results.append(state)
            ok_count += 1
            logger.info(
                "[CTA] %-8s %-25s  signal=%-12s  sharpe=%+.2f  ret12m=%+.1f%%",
                ticker, name, state["trend_signal"], state["sharpe_12m"],
                state["return_12m"] * 100,
            )
        except Exception as exc:
            logger.error("[CTA] Failed %s (%s): %s", ticker, name, exc, exc_info=True)
            results.append({"ticker": ticker, "name": name, "status": "error", "error": str(exc)})

    return {
        "assets":    results,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "status":    "ok" if ok_count > 0 else "degraded",
    }
