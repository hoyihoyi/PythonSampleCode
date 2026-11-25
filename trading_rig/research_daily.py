"""Swing research: every strategy x every instrument, split into
IN-SAMPLE (2000-2017, design) and OUT-OF-SAMPLE (2018-today, untouched holdout).
"""
import sys
import warnings

import pandas as pd

from data import load
from engine import backtest, combine, stats, vol_target
from instruments import INSTRUMENTS
from strategies import DAILY

warnings.filterwarnings("ignore")
IS_END, OOS_START = "2017-12-31", "2018-01-01"
SWING = [n for n in INSTRUMENTS if n != "WTI"]  # WTI: unadjusted futures roll gaps
EQUITY_IDX = ["US500", "NAS100", "US30", "GER40", "UK100", "JPN225"]


def run(strat, names, target=0.15, **kw):
    res = {}
    for n in names:
        df = load(n, "1d")
        f = DAILY[strat](df, **kw)
        pos = vol_target(f, df["Close"], target)
        res[n] = backtest(df, pos, n)
    return res


def split_stats(series, pos=None):
    a, b = series.loc[:IS_END], series.loc[OOS_START:]
    return stats(a), stats(b)


if __name__ == "__main__":
    strats = sys.argv[1:] or list(DAILY)
    rows = []
    for s in strats:
        names = EQUITY_IDX if s in ("rsi2_mr", "ibs_mr") else SWING
        res = run(s, names)
        for n, r in res.items():
            i, o = split_stats(r.ret)
            rows.append(dict(strat=s, inst=n, IS_sh=i["sharpe"], OOS_sh=o["sharpe"],
                             IS_gross=stats(r.gross.loc[:IS_END])["sharpe"],
                             OOS_gross=stats(r.gross.loc[OOS_START:])["sharpe"],
                             OOS_cagr=o["cagr"], OOS_dd=o["maxdd"],
                             cost_py=r.cost.loc[OOS_START:].mean() * 252))
        port = combine(res, target_vol=0.12)
        i, o = split_stats(port)
        rows.append(dict(strat=s, inst="PORTFOLIO", IS_sh=i["sharpe"], OOS_sh=o["sharpe"],
                         OOS_cagr=o["cagr"], OOS_dd=o["maxdd"], cost_py=float("nan")))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    t.to_csv("results/daily_grid.csv", index=False)
