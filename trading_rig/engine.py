"""Vectorised backtest engine for spread-betting style trading.

Conventions
-----------
* A strategy returns `pos`: target exposure, as a fraction of account equity
  in notional terms (1.0 = notional equal to equity, i.e. 1x leverage),
  decided with information up to and including the CLOSE of bar t.
* That position is filled at the OPEN of bar t+1 (no look-ahead).
* Bar return for t+1 = pos[t-1] * gap(close_t -> open_t+1)
                     + pos[t]   * intrabar(open_t+1 -> close_t+1)
* Costs: every change in position pays half spread + slippage (in price units,
  converted to return). Overnight financing is charged on |pos| for every
  calendar day the position is held over a day boundary (weekends included).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from instruments import INSTRUMENTS, INTRADAY_SPREAD_MULT


_RF = None


def benchmark() -> pd.Series:
    """Annualised benchmark rate (US 3m T-bill, FRED DTB3) as a decimal."""
    global _RF
    if _RF is None:
        from pathlib import Path
        f = Path(__file__).parent / "data" / "DTB3.csv"
        s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
        _RF = (pd.to_numeric(s, errors="coerce") / 100).ffill()
    return _RF


FX_CCY = {"EUR": "EZ", "GBP": "GB", "JPY": "JP", "AUD": "AU", "CAD": "CA",
          "CHF": "CH", "NZD": "NZ", "USD": "US"}
_RATES = {}


def ccy_rate(ccy: str) -> pd.Series:
    """3m interbank rate (FRED, monthly) as decimal, lagged 1 month for publication."""
    if ccy not in _RATES:
        from pathlib import Path
        f = Path(__file__).parent / "data" / f"rate_{FX_CCY[ccy]}.csv"
        s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
        s = (pd.to_numeric(s, errors="coerce") / 100).ffill()
        s.index = s.index + pd.offsets.MonthBegin(1)  # available a month later
        _RATES[ccy] = s.resample("D").ffill()
    return _RATES[ccy]


def fx_carry(pair: str) -> pd.Series:
    """Annual interest differential earned by a LONG position: r_base - r_quote."""
    b, q = pair[:3], pair[3:]
    s = ccy_rate(b).sub(ccy_rate(q)).dropna()
    return s


@dataclass
class Result:
    ret: pd.Series          # net returns per bar
    gross: pd.Series        # returns before costs
    pos: pd.Series          # executed position (fraction of equity)
    cost: pd.Series         # cost per bar (positive = drag)

    def stats(self, periods_per_year=252) -> dict:
        return stats(self.ret, self.pos, periods_per_year, self.gross, self.cost)


def backtest(df: pd.DataFrame, pos: pd.Series, name: str, intraday: bool = False,
             cost_mult: float = 1.0, buffer: float = 0.25) -> Result:
    spec = INSTRUMENTS[name]
    o, c = df["Open"], df["Close"]
    pos = pos.reindex(df.index).fillna(0.0).clip(-20, 20)
    if buffer:
        pos = apply_buffer(pos, buffer)

    held = pos.shift(1).fillna(0.0)      # position live during bar t (decided at t-1 close)
    held_prev = pos.shift(2).fillna(0.0) # position held across the gap into bar t
    gap = (o / c.shift(1) - 1).fillna(0.0)
    intra = (c / o - 1).fillna(0.0)
    gross = held_prev * gap + held * intra

    # transaction costs, paid at open of bar t when position changes
    spr = spec["spread"] * (INTRADAY_SPREAD_MULT if intraday else 1.0)
    # costs quoted in price units at TODAY's price level -> convert to a fraction
    # of price and apply that fraction throughout history
    ref = c.iloc[-1]
    per_unit = (spr / 2 + spec["slip"]) / ref * cost_mult
    turnover = (held - held_prev).abs()
    tcost = turnover * per_unit

    # financing: days crossed between previous bar and this bar's open
    idx = df.index
    days = pd.Series(idx.normalize(), index=idx).diff().dt.days.fillna(0).clip(lower=0)
    rf = benchmark().reindex(idx, method="ffill").fillna(0.0).values
    if spec["cls"] == "fx":
        diff = fx_carry(name).reindex(idx, method="ffill").fillna(0.0).values  # r_base - r_quote
        rate = np.where(held_prev > 0, spec["markup"] - diff, spec["markup"] + diff)
    elif spec["cls"] == "crypto":
        rate = np.full(len(idx), spec["markup"])
    elif spec["cls"] == "energy":   # futures-based SB: no overnight financing
        rate = np.zeros(len(idx))
    else:
        long_r = rf + spec["markup"] - spec["div"]
        short_r = spec["markup"] - rf + spec["div"]   # negative => short gets paid
        rate = np.where(held_prev > 0, long_r, short_r)
    fcost = held_prev.abs() * rate / 365.0 * days * cost_mult

    cost = tcost + fcost
    net = gross - cost
    return Result(net, gross, held, cost)


def apply_buffer(pos: pd.Series, buf: float) -> pd.Series:
    """Only trade when the target drifts more than `buf` (relative) from the
    current holding, or when the sign flips / goes flat. Cuts churn from
    vol-targeting without changing the signal."""
    tgt = pos.values
    out = np.zeros_like(tgt)
    cur = 0.0
    for i, t in enumerate(tgt):
        if t == 0 or cur == 0 or np.sign(t) != np.sign(cur) or abs(t - cur) > buf * abs(cur):
            cur = t
        out[i] = cur
    return pd.Series(out, index=pos.index)


def apply_buffer(pos: pd.Series, buf: float) -> pd.Series:
    """Only trade when the target drifts more than `buf` (relative) from the
    current holding, or when the sign flips / goes flat. Cuts churn from
    vol-targeting without changing the signal."""
    tgt = pos.values
    out = np.zeros_like(tgt)
    cur = 0.0
    for i, t in enumerate(tgt):
        if t == 0 or cur == 0 or np.sign(t) != np.sign(cur) or abs(t - cur) > buf * abs(cur):
            cur = t
        out[i] = cur
    return pd.Series(out, index=pos.index)


def stats(ret: pd.Series, pos=None, ppy=252, gross=None, cost=None) -> dict:
    ret = ret.dropna()
    if len(ret) < 20 or ret.std() == 0:
        return dict(sharpe=0.0, cagr=0.0, vol=0.0, maxdd=0.0, n=len(ret))
    eq = (1 + ret).cumprod()
    yrs = len(ret) / ppy
    dd = eq / eq.cummax() - 1
    out = dict(
        sharpe=ret.mean() / ret.std() * np.sqrt(ppy),
        cagr=eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0,
        vol=ret.std() * np.sqrt(ppy),
        maxdd=dd.min(),
        n=len(ret),
    )
    if pos is not None:
        out["turnover_py"] = pos.diff().abs().sum() / yrs
        out["exposure"] = (pos.abs() > 1e-9).mean()
    if gross is not None and gross.std() > 0:
        out["gross_sharpe"] = gross.mean() / gross.std() * np.sqrt(ppy)
        out["cost_drag_py"] = cost.sum() / yrs
    return out


# ---------- helpers used by strategies ----------

def ann_vol(c: pd.Series, span=36, ppy=252) -> pd.Series:
    """EWMA annualised volatility of close-to-close returns (lagged-safe: uses data to t)."""
    r = c.pct_change()
    v = r.ewm(span=span, min_periods=span).std() * np.sqrt(ppy)
    return v.clip(lower=v.rolling(ppy * 2, min_periods=20).median() * 0.3)


def vol_target(signal: pd.Series, c: pd.Series, target=0.15, ppy=252, span=36, cap=10.0):
    """Scale a forecast in [-1,1] so that a full signal runs at `target` annual vol."""
    return (signal * target / ann_vol(c, span, ppy)).clip(-cap, cap)


def combine(results: dict, weights: dict | None = None, target_vol=None, ppy=252,
            lookback=None) -> pd.Series:
    """Equal-weight (or weighted) portfolio of per-instrument net returns.
    If target_vol is given, scale the whole series by a *trailing* realised
    vol estimate (no look-ahead) to hit the target."""
    rets = pd.DataFrame({k: r.ret for k, r in results.items()})
    w = pd.Series(weights) if weights else pd.Series(1.0, index=rets.columns)
    w = w / w.sum()
    # instruments that have not started yet get zero weight, re-normalise per row
    live = rets.notna().astype(float) * w
    port = (rets.fillna(0) * w).sum(axis=1) / live.sum(axis=1).replace(0, np.nan)
    port = port.dropna()
    if target_vol:
        lb = lookback or ppy
        rv = port.ewm(span=lb // 2, min_periods=lb // 4).std().shift(1) * np.sqrt(ppy)
        lev = (target_vol / rv).clip(upper=5.0).fillna(1.0)
        port = port * lev
    return port
