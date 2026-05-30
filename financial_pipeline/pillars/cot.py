"""
Pillar 3: CFTC Commitment of Traders (COT) Tracker

Downloads bulk disaggregated futures CSV files directly from the CFTC public
site (no API key required) for commodity markets (Gold, Crude Oil) and the
Traders in Financial Futures (TFF) report for 10-Year Treasury futures.

For each market we:
    1. Extract the latest "Managed Money" (or Asset Manager) long/short contracts.
    2. Compute Net Positioning: Longs − Shorts.
    3. Calculate a rolling 52-week Z-score against the trailing history.
    4. Flag Z-scores beyond ±2.0 as historic positioning extremes.
"""
from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime, timezone
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── CFTC public bulk-download URLs ────────────────────────────────────────────
# Current-year rolling files (updated every Friday ~3:30 PM ET)
DISAGG_CURRENT_URL = "https://www.cftc.gov/dea/newcot/f_Disagg.zip"
TFF_CURRENT_URL    = "https://www.cftc.gov/dea/newcot/FinFutTxt.zip"

# Prior-year archives for sufficient Z-score history
DISAGG_HIST_URLS: list[str] = [
    "https://www.cftc.gov/files/dea/history/fut_disagg_txt_2024.zip",
    "https://www.cftc.gov/files/dea/history/fut_disagg_txt_2023.zip",
]
TFF_HIST_URLS: list[str] = [
    "https://www.cftc.gov/files/dea/history/fin_fut_txt_2024.zip",
    "https://www.cftc.gov/files/dea/history/fin_fut_txt_2023.zip",
]

# ── Column names (CFTC CSV schema) ────────────────────────────────────────────
DATE_COL   = "As_of_Date_In_Form_YYMMDD"
MARKET_COL = "Market_and_Exchange_Names"
# Disaggregated report (commodities)
DISAGG_LONG  = "M_Money_Positions_Long_All"
DISAGG_SHORT = "M_Money_Positions_Short_All"
# Traders in Financial Futures report
TFF_LONG  = "Asset_Mgr_Positions_Long_All"
TFF_SHORT = "Asset_Mgr_Positions_Short_All"

# ── Market name strings (exact CFTC identifiers, upper-cased for matching) ────
COMMODITY_TARGETS: dict[str, str] = {
    "Gold":      "GOLD - COMMODITY EXCHANGE INC.",
    "Crude Oil": "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
}
TFF_TARGETS: dict[str, str] = {
    "10Y Treasury": "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
}

REQUEST_TIMEOUT = 45  # seconds


# ── TypedDict schemas ─────────────────────────────────────────────────────────

class COTAsset(TypedDict):
    name: str
    as_of_date: str
    mm_longs: int
    mm_shorts: int
    mm_net: int
    zscore_52w: float | None
    is_extreme_long: bool
    is_extreme_short: bool


class COTResult(TypedDict):
    assets: list[Any]
    timestamp: str
    status: str


# ── IO helpers ────────────────────────────────────────────────────────────────

def _zip_url_to_df(url: str) -> pd.DataFrame:
    """Download a CFTC zip archive and parse the first CSV/TXT inside it."""
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        candidates = [n for n in zf.namelist() if n.lower().endswith((".txt", ".csv"))]
        if not candidates:
            raise ValueError(f"No text file inside zip from {url}")
        with zf.open(candidates[0]) as fh:
            return pd.read_csv(fh, low_memory=False)


def _load_combined(current_url: str, hist_urls: list[str]) -> pd.DataFrame:
    """
    Fetch current-year file plus as many prior-year archives as are reachable,
    concatenate, and return one big DataFrame for Z-score history.
    """
    frames: list[pd.DataFrame] = []
    for url in [current_url] + hist_urls:
        try:
            df = _zip_url_to_df(url)
            frames.append(df)
            logger.debug("[COT] Loaded %d rows from %s", len(df), url)
        except Exception as exc:
            logger.warning("[COT] Skipping %s: %s", url, exc)

    if not frames:
        raise RuntimeError("All CFTC download URLs failed — cannot build COT history.")

    return pd.concat(frames, ignore_index=True)


# ── Data parsing ──────────────────────────────────────────────────────────────

def _parse_yymmdd(val: Any) -> pd.Timestamp | pd.NaT:
    """Convert CFTC's YYMMDD integer (e.g. 230103) to a proper Timestamp."""
    try:
        s = str(int(val)).zfill(6)
        return pd.to_datetime(s, format="%y%m%d")
    except Exception:
        return pd.NaT


def _extract_market_series(
    df: pd.DataFrame,
    market_name: str,
    long_col: str,
    short_col: str,
) -> pd.DataFrame:
    """
    Filter rows matching a specific CFTC market name and return a clean
    time-series of (date, long, short, net) sorted chronologically.
    """
    mask    = df[MARKET_COL].astype(str).str.strip().str.upper() == market_name.upper()
    subset  = df.loc[mask, [DATE_COL, long_col, short_col]].copy()
    subset["date"]   = subset[DATE_COL].apply(_parse_yymmdd)
    subset[long_col]  = pd.to_numeric(subset[long_col],  errors="coerce").fillna(0)
    subset[short_col] = pd.to_numeric(subset[short_col], errors="coerce").fillna(0)
    subset["net"]     = subset[long_col] - subset[short_col]

    return (
        subset
        .dropna(subset=["date"])
        .drop_duplicates(subset="date")
        .sort_values("date")
        .reset_index(drop=True)
    )


# ── Z-score ───────────────────────────────────────────────────────────────────

def _zscore_52w(series: pd.Series) -> float | None:
    """
    Z-score of the most recent value relative to the trailing 52-observation
    window (weekly COT reports → 52 weeks ≈ 1 year).
    Returns None if fewer than 10 observations are available.
    """
    if len(series) < 10:
        return None
    window = series.iloc[-52:] if len(series) >= 52 else series
    mu  = float(window.mean())
    std = float(window.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        return None
    return float((float(series.iloc[-1]) - mu) / std)


# ── Per-market processing ─────────────────────────────────────────────────────

def _process_market(
    df_full: pd.DataFrame,
    name: str,
    market_name: str,
    long_col: str,
    short_col: str,
) -> COTAsset:
    ts = _extract_market_series(df_full, market_name, long_col, short_col)
    if ts.empty:
        raise ValueError(f"Zero rows matched market name: '{market_name}'")

    latest    = ts.iloc[-1]
    mm_longs  = int(latest[long_col])
    mm_shorts = int(latest[short_col])
    mm_net    = int(latest["net"])
    z         = _zscore_52w(ts["net"])

    return {
        "name":             name,
        "as_of_date":       str(latest["date"].date()),
        "mm_longs":         mm_longs,
        "mm_shorts":        mm_shorts,
        "mm_net":           mm_net,
        "zscore_52w":       round(z, 4) if z is not None else None,
        "is_extreme_long":  z is not None and z >  2.0,
        "is_extreme_short": z is not None and z < -2.0,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run() -> COTResult:
    """Execute Pillar 3: COT Positioning Tracker."""
    logger.info("[COT] Fetching CFTC Disaggregated (commodities) and TFF (financials) reports")
    results: list[Any] = []

    # ── Commodities: Gold, Crude Oil ──────────────────────────────────────────
    try:
        disagg_df = _load_combined(DISAGG_CURRENT_URL, DISAGG_HIST_URLS)
        for name, market_name in COMMODITY_TARGETS.items():
            try:
                asset = _process_market(disagg_df, name, market_name, DISAGG_LONG, DISAGG_SHORT)
                results.append(asset)
                z_label = f"{asset['zscore_52w']:.2f}" if asset["zscore_52w"] is not None else "N/A"
                logger.info(
                    "[COT] %-12s  net=%+10d  z=%s  %s",
                    name, asset["mm_net"], z_label,
                    "⚠ EXTREME" if asset["is_extreme_long"] or asset["is_extreme_short"] else "",
                )
            except Exception as exc:
                logger.error("[COT] %s processing failed: %s", name, exc, exc_info=True)
                results.append({"name": name, "status": "error", "error": str(exc)})
    except Exception as exc:
        logger.error("[COT] Could not load disaggregated report: %s", exc, exc_info=True)

    # ── Financial Futures: 10Y Treasury ──────────────────────────────────────
    try:
        tff_df = _load_combined(TFF_CURRENT_URL, TFF_HIST_URLS)
        for name, market_name in TFF_TARGETS.items():
            try:
                asset = _process_market(tff_df, name, market_name, TFF_LONG, TFF_SHORT)
                results.append(asset)
                z_label = f"{asset['zscore_52w']:.2f}" if asset["zscore_52w"] is not None else "N/A"
                logger.info(
                    "[COT] %-12s  net=%+10d  z=%s  %s",
                    name, asset["mm_net"], z_label,
                    "⚠ EXTREME" if asset["is_extreme_long"] or asset["is_extreme_short"] else "",
                )
            except Exception as exc:
                logger.error("[COT] %s processing failed: %s", name, exc, exc_info=True)
                results.append({"name": name, "status": "error", "error": str(exc)})
    except Exception as exc:
        logger.error("[COT] Could not load TFF report: %s", exc, exc_info=True)

    ok = [r for r in results if "mm_net" in r]
    return {
        "assets":    results,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "status":    "ok" if ok else "degraded",
    }
