"""Candidate production portfolio: diversified sleeves of the edges that
survived out-of-sample testing. All parameters are textbook defaults fixed
before the 2018+ holdout was examined (no tuning on OOS).

Sleeves
  A  index_pullback : RSI(2)<10 pullbacks in a bull regime (close > 200d), exit close > 5d SMA
  B  turn_of_month  : long indices last trading day -> 3rd trading day of new month
  C  fx_carry       : interest-differential carry, switched off in equity risk-off regimes
  D  index_regime   : vol-managed long index exposure, only in bull & non-crash vol regime
"""
import warnings

import numpy as np
import pandas as pd

from data import load
from engine import backtest, combine, stats, vol_target
from strategies import fx_carry_sig, rsi2_mr, turn_of_month, vol_regime_long

warnings.filterwarnings("ignore")

INDICES = ["US500", "NAS100", "US30", "US2000", "GER40", "FRA40", "EU50", "UK100",
           "JPN225", "HK50", "AUS200"]
FX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURJPY", "GBPJPY", "EURGBP",
      "AUDJPY", "NZDUSD", "USDCHF", "EURCHF"]

SLEEVES = {
    "A_index_pullback": (INDICES, lambda df, n: rsi2_mr(df, lo=10, trend_n=200, exit_n=5)),
    "B_turn_of_month": ([i for i in INDICES if i != "UK100"], lambda df, n: turn_of_month(df)),
    "C_fx_carry": (FX, lambda df, n: fx_carry_sig(df, n)),
    "D_index_regime": (INDICES, lambda df, n: vol_regime_long(df)),
}
SLEEVE_VOL = 0.12


CALIB_END = "2017-12-31"   # leverage constants are calibrated on in-sample data only


def sleeve(name, cost_mult=1.0, start="2003-06-01", per_inst_target=0.15):
    """Equal-weight the instruments; then apply ONE constant leverage factor so
    the sleeve ran at SLEEVE_VOL in-sample. (Dynamic re-levering of sparse
    strategies from trailing vol inflates drawdowns, so we don't do it.)"""
    names, fn = SLEEVES[name]
    res = {}
    for n in names:
        df = load(n, "1d", start)
        res[n] = backtest(df, vol_target(fn(df, n), df["Close"], per_inst_target), n, cost_mult=cost_mult)
    raw = combine(res)
    k = SLEEVE_VOL / (raw.loc[:CALIB_END].std() * np.sqrt(252))
    return raw * k, res


def portfolio(sleeves=None, cost_mult=1.0, target=0.12, weights=None):
    sleeves = sleeves or list(SLEEVES)
    rets = pd.DataFrame({s: sleeve(s, cost_mult)[0] for s in sleeves}).dropna(how="all").fillna(0)
    w = pd.Series(weights or {s: 1.0 for s in sleeves})
    w = w / w.sum()
    port = (rets * w).sum(axis=1)
    k = target / (port.loc["2004-06-01":CALIB_END].std() * np.sqrt(252))
    port = port * k
    return port.loc["2004-06-01":], rets.loc["2004-06-01":]


def report(p, label):
    i, o, a = stats(p.loc[:"2017"]), stats(p.loc["2018":]), stats(p)
    print(f"{label:34s} IS {i['sharpe']:5.2f}  OOS {o['sharpe']:5.2f} | OOS CAGR {o['cagr']:6.1%}"
          f"  OOS maxDD {o['maxdd']:6.1%} | full {a['sharpe']:.2f}")


if __name__ == "__main__":
    port, rets = portfolio()
    for s in rets:
        report(rets[s], s)
    print("\nsleeve correlations (full period):")
    print(rets.corr().round(2))
    print()
    report(port, "ENSEMBLE A+B+C+D @12% vol")
    p3, _ = portfolio(["A_index_pullback", "B_turn_of_month", "C_fx_carry"])
    report(p3, "ENSEMBLE A+B+C (no beta sleeve)")
    for cm in (1.5, 2.0):
        pc, _ = portfolio(cost_mult=cm)
        report(pc, f"ENSEMBLE costs x{cm}")
    bh, _ = sleeve("D_index_regime")
    yr = port.groupby(port.index.year).apply(lambda x: (1 + x).prod() - 1)
    print("\ncalendar-year returns (ensemble, 12% vol target):")
    print((yr * 100).round(1).to_string())
    port.to_csv("results/ensemble_daily.csv")
    rets.to_csv("results/sleeves_daily.csv")
