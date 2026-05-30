"""
Pillar 1: Options Microstructure & Dealer Gamma Exposure (GEX)

Fetches the full SPY options chain via yfinance, runs a Black-Scholes gamma
engine over every (strike, expiry, vol) tuple, and aggregates Net Dealer GEX
by strike under the assumption:
    • Dealers are LONG calls  → positive gamma contribution
    • Dealers are SHORT puts  → negative gamma contribution

Dollar GEX per row: Γ × OI × Spot² × 0.01
(The 0.01 factor converts to "dollars moved per 1% spot move × 100-share contract.")
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TypedDict

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

logger = logging.getLogger(__name__)

TICKER = "SPY"
RISK_FREE_RATE = 0.05  # ~5% annualised; update via env var if desired


# ── TypedDict schemas ─────────────────────────────────────────────────────────

class StrikeGEX(TypedDict):
    strike: float
    net_dollar_gex: float
    call_dollar_gex: float
    put_dollar_gex: float
    call_oi: int
    put_oi: int


class GEXResult(TypedDict):
    ticker: str
    spot_price: float
    call_wall: float
    put_wall: float
    gex_flip_zone: float | None
    top_5_strikes: list[StrikeGEX]
    total_net_gex: float
    timestamp: str
    status: str


# ── Black-Scholes engine ──────────────────────────────────────────────────────

def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes Gamma: φ(d₁) / (S × σ × √T)
    Both calls and puts share the same gamma formula.
    Returns 0.0 for degenerate inputs (expired, zero vol, non-positive prices).
    """
    if T <= 1e-7 or sigma <= 1e-7 or S <= 0.0 or K <= 0.0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def _time_to_expiry_years(expiry_str: str) -> float:
    """Days to expiry as a fraction of a 365-day year; floored at 0."""
    expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    days = (expiry_dt - now).total_seconds() / 86_400.0
    return max(days / 365.0, 0.0)


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_full_chain(ticker: yf.Ticker) -> pd.DataFrame:
    """
    Pull every available expiration from yfinance and stack calls + puts
    into a single DataFrame with a [T, option_type] column appended.
    """
    expirations = ticker.options
    if not expirations:
        raise ValueError(f"yfinance returned no option expirations for {TICKER}")

    frames: list[pd.DataFrame] = []
    for exp in expirations:
        T = _time_to_expiry_years(exp)
        if T <= 0.0:
            continue
        try:
            chain = ticker.option_chain(exp)
        except Exception as exc:
            logger.warning("[GEX] Skipping expiry %s — fetch error: %s", exp, exc)
            continue

        calls = chain.calls.assign(option_type="call", expiry=exp, T=T)
        puts  = chain.puts.assign(option_type="put",  expiry=exp, T=T)
        frames.extend([calls, puts])

    if not frames:
        raise ValueError("All option expirations are expired or failed to fetch.")

    df = pd.concat(frames, ignore_index=True)

    required = {"strike", "openInterest", "impliedVolatility", "option_type", "T"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Options chain missing required columns: {missing}")

    return df


# ── GEX calculation ───────────────────────────────────────────────────────────

def _apply_gex(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """
    Vectorised calculation of per-row gamma and dollar GEX.
    Dealer convention applied here:
        calls → +dollar_gex  (dealer long calls = long gamma)
        puts  → -dollar_gex  (dealer short puts = short gamma)
    """
    df = df.copy()
    df["openInterest"]      = pd.to_numeric(df["openInterest"],      errors="coerce").fillna(0.0)
    df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce").fillna(0.0)

    df["gamma"] = df.apply(
        lambda r: _bs_gamma(
            S=spot, K=r["strike"], T=r["T"],
            r=RISK_FREE_RATE, sigma=r["impliedVolatility"],
        ),
        axis=1,
    )

    # Dollar Gamma = Γ × OI × S² × 0.01
    df["dollar_gex"] = df["gamma"] * df["openInterest"] * (spot ** 2) * 0.01

    # Net dealer gamma: positive for calls, negative for puts
    df["net_dollar_gex"] = np.where(df["option_type"] == "call", df["dollar_gex"], -df["dollar_gex"])
    return df


def _aggregate_by_strike(gex_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse all expirations into a single aggregate GEX per strike."""
    def _agg(g: pd.DataFrame) -> pd.Series:
        call_mask = g["option_type"] == "call"
        put_mask  = ~call_mask
        return pd.Series({
            "net_dollar_gex":  g["net_dollar_gex"].sum(),
            "call_dollar_gex": g.loc[call_mask, "dollar_gex"].sum(),
            "put_dollar_gex":  g.loc[put_mask,  "dollar_gex"].sum(),
            "call_oi":         int(g.loc[call_mask, "openInterest"].sum()),
            "put_oi":          int(g.loc[put_mask,  "openInterest"].sum()),
        })

    return gex_df.groupby("strike").apply(_agg).reset_index()


# ── Level identification ──────────────────────────────────────────────────────

def _find_call_wall(by_strike: pd.DataFrame) -> float:
    """Strike with the highest positive net GEX — the strongest overhead resistance."""
    positive = by_strike[by_strike["net_dollar_gex"] > 0]
    if positive.empty:
        return float(by_strike["strike"].iloc[len(by_strike) // 2])
    return float(positive.loc[positive["net_dollar_gex"].idxmax(), "strike"])


def _find_put_wall(by_strike: pd.DataFrame) -> float:
    """Strike with the most negative net GEX — the strongest downside support."""
    negative = by_strike[by_strike["net_dollar_gex"] < 0]
    if negative.empty:
        return float(by_strike["strike"].iloc[0])
    return float(negative.loc[negative["net_dollar_gex"].idxmin(), "strike"])


def _find_gex_flip(by_strike: pd.DataFrame) -> float | None:
    """
    Strike where cumulative GEX (sorted ascending) crosses zero.
    Below this level dealers are net short gamma (amplify moves);
    above it they are net long gamma (dampen moves).
    """
    sorted_df = by_strike.sort_values("strike")
    cumgex    = sorted_df["net_dollar_gex"].cumsum().values
    crosses   = np.where(np.diff(np.sign(cumgex)))[0]
    if crosses.size == 0:
        return None
    flip_idx = crosses[0]
    return float(sorted_df.iloc[flip_idx]["strike"])


# ── Public entry point ────────────────────────────────────────────────────────

def run() -> GEXResult:
    """Execute Pillar 1 and return a structured GEXResult dict."""
    logger.info("[GEX] Fetching %s spot price and full options chain", TICKER)

    ticker = yf.Ticker(TICKER)
    hist   = ticker.history(period="2d")
    if hist.empty:
        raise RuntimeError(f"Cannot fetch spot price for {TICKER}")
    spot = float(hist["Close"].iloc[-1])
    logger.info("[GEX] Spot: %.2f", spot)

    chain_df = _fetch_full_chain(ticker)
    logger.info("[GEX] Chain rows: %d across %d expirations",
                len(chain_df), chain_df["expiry"].nunique())

    gex_df      = _apply_gex(chain_df, spot)
    by_strike   = _aggregate_by_strike(gex_df)

    call_wall   = _find_call_wall(by_strike)
    put_wall    = _find_put_wall(by_strike)
    flip_zone   = _find_gex_flip(by_strike)
    total_gex   = float(by_strike["net_dollar_gex"].sum())

    top_rows    = by_strike.nlargest(5, by_strike["net_dollar_gex"].abs().rename("abs_gex")
                                       if False else "net_dollar_gex")
    # Use abs magnitude for top-5 selection
    top_5_idx   = by_strike["net_dollar_gex"].abs().nlargest(5).index
    top_5_df    = by_strike.loc[top_5_idx].sort_values("net_dollar_gex", ascending=False)

    top_5: list[StrikeGEX] = [
        {
            "strike":         float(row["strike"]),
            "net_dollar_gex": round(float(row["net_dollar_gex"]), 2),
            "call_dollar_gex": round(float(row["call_dollar_gex"]), 2),
            "put_dollar_gex":  round(float(row["put_dollar_gex"]),  2),
            "call_oi":        int(row["call_oi"]),
            "put_oi":         int(row["put_oi"]),
        }
        for _, row in top_5_df.iterrows()
    ]

    logger.info("[GEX] Call Wall=%.0f | Put Wall=%.0f | Flip=%.0f | NetGEX=%.0f",
                call_wall, put_wall, flip_zone or 0.0, total_gex)

    return {
        "ticker":        TICKER,
        "spot_price":    round(spot, 2),
        "call_wall":     call_wall,
        "put_wall":      put_wall,
        "gex_flip_zone": flip_zone,
        "top_5_strikes": top_5,
        "total_net_gex": round(total_gex, 2),
        "timestamp":     datetime.now(tz=timezone.utc).isoformat(),
        "status":        "ok",
    }
