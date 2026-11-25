"""Intraday (hourly) research: session-aware engine + strategies taken from
published research with FIXED (pre-registered) parameters. Only ~2 years of
hourly data are available from Yahoo, so nothing here is tuned.
"""
import warnings

import numpy as np
import pandas as pd

from data import load
from engine import Result, stats, ann_vol
from instruments import INSTRUMENTS, INTRADAY_SPREAD_MULT

warnings.filterwarnings("ignore")

TZ = {"US500": "America/New_York", "NAS100": "America/New_York", "US30": "America/New_York",
      "US2000": "America/New_York", "GER40": "Europe/Berlin", "FRA40": "Europe/Paris",
      "EU50": "Europe/Berlin", "UK100": "Europe/London", "JPN225": "Asia/Tokyo",
      "HK50": "Asia/Hong_Kong", "AUS200": "Australia/Sydney"}


def sessionize(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Add local time, session date, bar number in session, last-bar flag,
    previous session close."""
    tz = TZ.get(name, "UTC")
    loc = df.index.tz_localize("UTC").tz_convert(tz)
    df = df.copy()
    df["hour"] = loc.hour + loc.minute / 60
    df["sdate"] = loc.date
    g = df.groupby("sdate")
    df["bar"] = g.cumcount()
    df["nbar"] = g["Close"].transform("size")
    df["last"] = df["bar"] == df["nbar"] - 1
    sc = g["Close"].last()
    df["prev_close"] = df["sdate"].map(sc.shift(1))
    so = g["Open"].first()
    df["sess_open"] = df["sdate"].map(so)
    return df


def backtest_intraday(df, pos, name, moc=True, cost_mult=1.0, daily_vol_lookback=20):
    """pos decided at bar close, executed at next bar open; if moc and the bar
    is the session's last bar, execution happens at that bar's close
    (market-on-close; SB index markets keep trading after the cash close)."""
    spec = INSTRUMENTS[name]
    o, c = df["Open"], df["Close"]
    pos = pos.reindex(df.index).fillna(0.0)
    held = pos.shift(1).fillna(0.0)
    held_prev = pos.shift(2).fillna(0.0)
    last_prev = df["last"].shift(1).fillna(False).astype(bool)
    gap_pos = np.where(last_prev & moc, held, held_prev)   # MOC: new pos already on at prev close
    gap = (o / c.shift(1) - 1).fillna(0.0)
    intra = (c / o - 1).fillna(0.0)
    gross = gap_pos * gap + held * intra
    per_unit = (spec["spread"] * INTRADAY_SPREAD_MULT / 2 + spec["slip"]) / c.iloc[-1] * cost_mult
    turnover = (held - held_prev).abs()
    tcost = turnover * per_unit
    # financing on positions carried across a calendar day
    days = pd.Series(df.index.normalize(), index=df.index).diff().dt.days.fillna(0).clip(lower=0)
    fcost = pd.Series(np.abs(gap_pos), index=df.index) * 0.07 / 365 * days
    cost = tcost + fcost
    return Result(gross - cost, gross, held, cost)


def daily_returns(r: Result) -> pd.Series:
    """Aggregate bar returns to calendar days (for comparable Sharpe ratios)."""
    return (1 + r.ret).groupby(r.ret.index.normalize()).prod() - 1


def sized(df, sig, target=0.15):
    """Size intraday trades by DAILY vol so each day carries similar risk."""
    dc = df.groupby("sdate")["Close"].last()
    dv = ann_vol(dc, span=20).shift(1)  # yesterday's estimate
    v = df["sdate"].map(dv).astype(float)
    return (sig * target / v).clip(-10, 10).fillna(0)


# ---------------- strategies ----------------

def last_hour_momentum(df):
    """Market intraday momentum (Gao-Han-Li-Zhou 2018; Baltussen-Da-Lammers-Martens
    2021): return from prior close to the start of the final bar predicts the final
    bar's direction (dealer gamma hedging / late rebalancing). Signal at close of
    the 2nd-last bar, hold the last bar, exit on the close."""
    r = df["Close"] / df["prev_close"] - 1
    sig = np.sign(r).where(df["bar"] == df["nbar"] - 2, 0.0)
    return sig.fillna(0)


def opening_range_breakout(df, n_open=1):
    """First-hour range breakout, held to the session close; flat if price
    closes back inside the far side of the range (stop)."""
    g = df.groupby("sdate")
    first = df["bar"] < n_open
    hi = df["High"].where(first).groupby(df["sdate"]).transform("max")
    lo = df["Low"].where(first).groupby(df["sdate"]).transform("min")
    c = df["Close"].values
    pos = np.zeros(len(df))
    bar, last, hv, lv = df["bar"].values, df["last"].values, hi.values, lo.values
    for i in range(len(df)):
        if bar[i] < n_open or last[i]:
            pos[i] = 0
            continue
        p = pos[i - 1] if bar[i] > n_open else 0
        if bar[i] == n_open:
            traded = False
        if p == 0 and not traded:
            if c[i] > hv[i]:
                p, traded = 1, True
            elif c[i] < lv[i]:
                p, traded = -1, True
        elif p == 1 and c[i] < lv[i]:
            p = 0
        elif p == -1 and c[i] > hv[i]:
            p = 0
        pos[i] = p
    return pd.Series(pos, index=df.index)


def gap_fade(df, k=0.5):
    """Fade opening gaps larger than k x daily sigma; enter after the first bar,
    exit when the gap is filled or at the close."""
    dc = df.groupby("sdate")["Close"].last()
    dsig = dc.pct_change().rolling(20).std().shift(1)
    sig_ = df["sdate"].map(dsig)
    gap = df["sess_open"] / df["prev_close"] - 1
    c = df["Close"].values
    pc = df["prev_close"].values
    big = (gap.abs() > k * sig_).values
    d = -np.sign(gap.values)
    pos = np.zeros(len(df))
    bar, last = df["bar"].values, df["last"].values
    for i in range(len(df)):
        if bar[i] == 0 or last[i]:
            continue
        if bar[i] == 1 or pos[i - 1] != 0:  # enter at close of first bar, keep until filled
            if big[i] and ((d[i] > 0 and c[i] < pc[i]) or (d[i] < 0 and c[i] > pc[i])):
                pos[i] = d[i]
    return pd.Series(pos, index=df.index)


def london_breakout(df, asia_end=7, entry_end=16, exit_hour=20):
    """FX/gold: Asian-session range (00:00-asia_end UTC) breakout during London/NY.
    One trade per day, stop on opposite side of range, exit at exit_hour UTC."""
    h = df.index.hour
    day = df.index.normalize()
    asia = h < asia_end
    hi = df["High"].where(asia).groupby(day).transform("max")
    lo = df["Low"].where(asia).groupby(day).transform("min")
    c = df["Close"].values
    hi, lo = hi.values, lo.values
    pos = np.zeros(len(df))
    traded = False
    for i in range(len(df)):
        if i == 0 or day[i] != day[i - 1]:
            traded = False
        p = pos[i - 1] if i and day[i] == day[i - 1] else 0
        if h[i] < asia_end or h[i] >= exit_hour:
            pos[i] = 0
            continue
        if p == 0 and not traded and h[i] < entry_end:
            if c[i] > hi[i]:
                p, traded = 1, True
            elif c[i] < lo[i]:
                p, traded = -1, True
        elif p == 1 and c[i] < lo[i]:
            p = 0
        elif p == -1 and c[i] > hi[i]:
            p = 0
        pos[i] = p
    return pd.Series(pos, index=df.index)


def hourly_reversal(df, z_entry=2.0, n=20):
    """Intraday liquidity-imbalance fade: when the last n hourly closes are
    stretched > z_entry sigmas, fade; exit at mean or session end."""
    c = df["Close"]
    z = ((c - c.rolling(n).mean()) / c.rolling(n).std()).values
    pos = np.zeros(len(df))
    last = df["last"].values
    for i in range(1, len(df)):
        p = pos[i - 1]
        if last[i]:
            pos[i] = 0
            continue
        if p == 0:
            p = 1 if z[i] < -z_entry else (-1 if z[i] > z_entry else 0)
        elif (p == 1 and z[i] > 0) or (p == -1 and z[i] < 0):
            p = 0
        pos[i] = p
    return pd.Series(pos, index=df.index)


IDX_US = ["US500", "NAS100", "US30", "US2000"]
IDX_EU = ["GER40", "FRA40", "EU50", "UK100"]
IDX_AS = ["JPN225", "HK50", "AUS200"]
FXM = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURJPY", "GBPJPY", "XAUUSD", "XAGUSD"]

TESTS = {
    "last_hour_mom": (last_hour_momentum, IDX_US + IDX_EU + IDX_AS),
    "orb": (opening_range_breakout, IDX_US + IDX_EU + IDX_AS),
    "gap_fade": (gap_fade, IDX_US + IDX_EU + IDX_AS),
    "london_bo": (london_breakout, FXM),
    "hourly_rev": (hourly_reversal, IDX_US + IDX_EU + FXM),
}


def run(test, names=None, cost_mult=1.0):
    fn, default = TESTS[test]
    out = {}
    for n in names or default:
        df = sessionize(load(n, "1h", None), n)
        pos = sized(df, fn(df))
        out[n] = backtest_intraday(df, pos, n, cost_mult=cost_mult)
    return out


if __name__ == "__main__":
    import sys
    for t in (sys.argv[1:] or TESTS):
        res = run(t)
        rows = []
        for n, r in res.items():
            d = daily_returns(r)
            g = (1 + r.gross).groupby(r.gross.index.normalize()).prod() - 1
            s = stats(d)
            rows.append(dict(inst=n, sharpe=s["sharpe"], gross_sharpe=stats(g)["sharpe"],
                             cagr=s["cagr"], maxdd=s["maxdd"],
                             trades_py=(r.pos.diff().abs() > 0).sum() / 2 / (len(d) / 252)))
        port = pd.concat({n: daily_returns(r) for n, r in res.items()}, axis=1).fillna(0).mean(axis=1)
        sp = stats(port)
        print(f"\n== {t}: portfolio net sharpe {sp['sharpe']:.2f}  cagr {sp['cagr']:.3f} dd {sp['maxdd']:.3f}")
        print(pd.DataFrame(rows).round(2).to_string(index=False))


def london_breakout_nr(df, asia_end=7, entry_end=16, exit_hour=20, lookback=20):
    """Regime-filtered variant: only trade when today's Asian range is narrower
    than its 20-day median (volatility compression -> expansion regime)."""
    base = london_breakout(df, asia_end, entry_end, exit_hour)
    day = df.index.normalize()
    asia = df.index.hour < asia_end
    rng = (df["High"].where(asia).groupby(day).max() - df["Low"].where(asia).groupby(day).min())
    med = rng.rolling(lookback, min_periods=10).median().shift(1)
    ok = (rng < med).reindex(day).values
    return base * ok


def orb_nr(df, lookback=20):
    """ORB only when the first-hour range is narrower than its 20-session median."""
    base = opening_range_breakout(df)
    first = df["bar"] == 0
    rng = (df["High"] - df["Low"]).where(first).groupby(df["sdate"]).max()
    med = rng.rolling(lookback, min_periods=10).median().shift(1)
    ok = df["sdate"].map(rng < med).astype(bool).values
    return base * ok


TESTS["london_bo_nr"] = (london_breakout_nr, FXM)
TESTS["orb_nr"] = (orb_nr, IDX_US + IDX_EU + IDX_AS)
