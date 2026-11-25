"""Download & cache OHLC data from Yahoo Finance.

    python data.py            # refresh everything
Daily: full history. Hourly: last ~730 days (Yahoo limit).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from instruments import INSTRUMENTS

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df = df[(df > 0).all(axis=1)]
    # repair occasional bad prints: High/Low must bracket Open/Close
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df[~df.index.duplicated()].sort_index()


def download(name: str, interval: str = "1d") -> pd.DataFrame:
    tk = INSTRUMENTS[name]["yf"]
    kw = dict(period="max") if interval == "1d" else dict(period="729d")
    for attempt in range(4):
        try:
            df = yf.download(tk, interval=interval, auto_adjust=False, progress=False, **kw)
            if len(df):
                return _clean(df)
        except Exception as e:  # network hiccup
            print("retry", name, e)
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"no data for {name}")


def despike(df: pd.DataFrame, k: float = 8.0) -> pd.DataFrame:
    """Remove one-bar spikes: a close that jumps > k robust-sigmas away and then
    comes straight back (>70% reversal next bar). Real breaks (e.g. SNB 2015)
    don't reverse and are kept."""
    df = df.copy()
    lc = np.log(df["Close"])
    r = lc.diff()
    sig = r.abs().rolling(100, min_periods=20).median().shift(1) * 1.4826
    bad = (r.abs() > k * sig) & (-r.shift(-1) * np.sign(r) > 0.7 * r.abs())
    for col in ("Open", "High", "Low", "Close"):
        df.loc[bad, col] = np.nan
    df = df.ffill()
    # opens that are themselves spikes vs both neighbouring closes
    go = np.log(df["Open"] / df["Close"].shift(1)).abs()
    gc = np.log(df["Open"] / df["Close"]).abs()
    badopen = (go > k * sig) & (gc > k * sig)
    df.loc[badopen, "Open"] = df["Close"].shift(1)[badopen]
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
    return df


def load(name: str, interval: str = "1d", start: str | None = "2000-01-01") -> pd.DataFrame:
    f = DATA / f"{name}_{interval}.csv"
    if not f.exists():
        download(name, interval).to_csv(f)
    df = despike(pd.read_csv(f, index_col=0, parse_dates=True))
    return df.loc[start:] if start else df


if __name__ == "__main__":
    for n in INSTRUMENTS:
        for iv in ("1d", "1h"):
            df = download(n, iv)
            df.to_csv(DATA / f"{n}_{iv}.csv")
            print(f"{n:7s} {iv}: {len(df):6d} bars {df.index[0].date()} -> {df.index[-1].date()}")
            time.sleep(1)
