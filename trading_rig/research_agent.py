"""Research agent: one command that refreshes data, re-runs every test, applies
pass/fail gates and writes results/REPORT.md. Run it monthly (maintenance).

    python research_agent.py            # use cached data
    python research_agent.py --refresh  # re-download from Yahoo / FRED first

Gates a strategy must pass before it is allowed near real money:
  G1  out-of-sample (2018+) Sharpe after costs  >= 0.30
  G2  in-sample Sharpe after costs              >= 0.30 (edge existed before too)
  G3  still profitable with costs x1.5
  G4  Deflated Sharpe probability               >= 0.90 (not explained by luck
      given the number of variants tried)
Also a drift monitor: flags a live sleeve whose last-6-month return is in the
worst 5% of its own history (time to review, not necessarily to stop).
"""
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import ensemble as E
from engine import stats
from validate import block_bootstrap_sharpe, deflated_sharpe

ROOT = Path(__file__).parent
N_TRIALS = 60  # running count of variants examined in this research (keep honest!)


def refresh():
    subprocess.run([sys.executable, str(ROOT / "data.py")], check=True, cwd=ROOT)
    import urllib.request
    ids = {"DTB3": "DTB3", **{f"rate_{c}": f"IR3TIB01{c}M156N" for c in
                               ["EZ", "GB", "JP", "AU", "CA", "CH", "NZ", "US"]}}
    for f, sid in ids.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        urllib.request.urlretrieve(url, ROOT / "data" / f"{f}.csv")


def gate_row(name, r, r_cost15):
    i, o = stats(r.loc[:"2017"]), stats(r.loc["2018":])
    dsr, hurdle = deflated_sharpe(r, N_TRIALS, 0.30)
    ci, _ = block_bootstrap_sharpe(r.loc["2018":], n_boot=500)
    g = dict(G1=o["sharpe"] >= 0.30, G2=i["sharpe"] >= 0.30,
             G3=stats(r_cost15)["sharpe"] > 0, G4=dsr >= 0.90)
    return dict(strategy=name, IS_sharpe=round(i["sharpe"], 2), OOS_sharpe=round(o["sharpe"], 2),
                OOS_CI=f"[{ci[0]:.2f}, {ci[2]:.2f}]", OOS_cagr=f"{o['cagr']:.1%}",
                OOS_maxdd=f"{o['maxdd']:.1%}", DSR=f"{dsr:.0%}", **{k: "PASS" if v else "fail" for k, v in g.items()},
                verdict="DEPLOYABLE" if all(g.values()) else ("WATCHLIST" if g["G1"] and g["G3"] else "REJECT"))


def drift(r: pd.Series):
    roll = (1 + r).rolling(126).apply(np.prod, raw=True) - 1
    last = roll.iloc[-1]
    pct = (roll.dropna() < last).mean()
    return last, pct


def main():
    if "--refresh" in sys.argv:
        refresh()
    rows, drift_rows = [], []
    for s in E.SLEEVES:
        r, _ = E.sleeve(s)
        r15, _ = E.sleeve(s, cost_mult=1.5)
        r, r15 = r.loc["2004-06-01":], r15.loc["2004-06-01":]
        rows.append(gate_row(s, r, r15))
        last, pct = drift(r)
        drift_rows.append(dict(sleeve=s, last_6m=f"{last:.1%}", percentile=f"{pct:.0%}",
                               flag="REVIEW" if pct < 0.05 else "ok"))
    port, _ = E.portfolio()
    p15, _ = E.portfolio(cost_mult=1.5)
    rows.append(gate_row("ENSEMBLE", port, p15))
    t = pd.DataFrame(rows)
    d = pd.DataFrame(drift_rows)
    md = [f"# Research agent report - {date.today()}", "",
          f"Variants examined to date (for deflated Sharpe): {N_TRIALS}", "",
          "## Strategy gates", "", t.to_markdown(index=False), "",
          "## Drift monitor (last 6 months vs own history)", "", d.to_markdown(index=False), ""]
    (ROOT / "results" / "REPORT.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
