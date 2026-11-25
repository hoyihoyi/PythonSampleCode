# Trading research rig — Pepperstone spread betting via cTrader

A backtesting and research rig for automated **swing** (daily) and **day-trading**
(hourly) strategies, with a cost model for Pepperstone spread betting (spread,
slippage, and overnight financing at benchmark ± markup with dividend
adjustments and real FX interest differentials). It also includes
cTrader cBots for the candidates that came closest.

## Bottom line (research run 2026-11-25)

**No strategy found here clears the bar for real money yet.** The best result is a
diversified ensemble that *is* profitable after all costs (2004–2026 Sharpe 0.46,
2018–2026 out-of-sample Sharpe 0.30, about 3–4% a year at 12% volatility). It
does **not** pass the deflated-Sharpe test: after ~60 variants were tried, that
performance is within what luck alone could produce. It also barely beats simply
holding indices, and it had a −36% drawdown.

| Family | Result after costs | Verdict |
|---|---|---|
| Daily trend following (EMA cross, TSMOM, Donchian, ER-regime switch), 28 markets | IS Sharpe ~0.5, **OOS 2018+ negative** (even gross ≈ 0) | Reject |
| Adaptive trend⇄mean-reversion switch (efficiency-ratio regime) | IS 0.3 → OOS −0.3 | Reject |
| Markov regime-switching (walk-forward) index timing | Cuts drawdown (−36% → −27%) but no Sharpe gain vs holding | Risk tool, not alpha |
| RSI(2) pullback in equity indices | IS 0.6 → OOS 0.1–0.25; loses on the first leg of crashes (Feb 2018, Feb 2020) | Edge decayed |
| Turn-of-month (indices) | 0.19 → 0.16, stable but small | Weak |
| Overnight drift (US indices) | Gross OOS Sharpe 1.3 (!), net 0.3 — spread and financing eat it | Thin |
| **FX carry + equity risk-off switch** | IS 0.18 → **OOS 0.36**; regime switch turned 2008's −27% into ~0% | **Watchlist (best)** |
| Index pairs mean reversion | Negative | Reject |
| Intraday: last-hour momentum, gap fade, hourly reversal | Negative even gross (2024–26) | Reject |
| Intraday: opening-range / Asian-range breakout | **Gross positive on all 9 FX/metal markets**, negative after conservative costs | Needs real tick data |

### Why it's hard on spread bets
* Holding costs: longs pay benchmark + ~2.5% a year on the **full notional**. With
  leverage this is a large drag on anything held for weeks, so even buy-and-hold indices only reach a Sharpe of about 0.3–0.4.
* Leverage does not create edge. It scales returns *and* costs. A pro account only
  lowers margin; risk is set by position size.
* Free data limits: hourly bars cover only ~2 years, too short to validate
  day-trading strategies (a 2-year Sharpe has a standard error of about ±0.7).

## Recommended next steps
1. **Re-test on Pepperstone's own data.** Load the cBots into cTrader Automate and run
   the built-in backtester with *tick data*. This uses Pepperstone's real spreads and
   swaps and gives 10+ years of intraday data, which this container couldn't download.
   `SessionBreakout` is the most important candidate to settle this way.
2. **Demo-forward-test** `FxCarryRegime` (and optionally `IndexRegimeSwing`) for
   3 months on a demo account before any live money.
3. Only go live with a strategy that passes all four gates in `research_agent.py`.
   Start at small size; the cBots have drawdown kill-switches.

## Layout
```
instruments.py     universe + Pepperstone SB cost model
data.py            Yahoo download/cache + spike cleaning   (python data.py)
engine.py          vectorised engine: close-signal -> next-open fill, spread/slip,
                   financing (FRED T-bill benchmark, FX rate differentials), buffering
strategies.py      strategy library (trend, breakout, mean reversion, regime, carry, pairs, HMM)
research_daily.py  swing grid: every strategy x instrument, IS 2000-17 vs OOS 2018+
intraday.py        session-aware hourly engine + intraday strategies
ensemble.py        candidate portfolio + cost stress
validate.py        block bootstrap CI + deflated Sharpe ratio
research_agent.py  one-shot pipeline + gates + drift monitor -> results/REPORT.md
ctrader/           cBots (.cs source + compiled .algo, built against cTrader.Automate 1.0.21)
```

## Running
```bash
pip install yfinance pandas numpy scipy statsmodels tabulate
python research_agent.py --refresh    # monthly maintenance run
python research_daily.py ewmac rsi2_mr   # ad-hoc grid
python intraday.py london_bo             # ad-hoc intraday test
```
Methodology rules used throughout: signals only use data up to the bar close; fills at
the next open; parameters fixed from the literature or chosen in-sample only; 2018+ is
held out; every variant counts towards the deflated-Sharpe trial count (`N_TRIALS`).

## cBots
| Bot | Chart | What it does |
|---|---|---|
| `FxCarryRegime` | Daily, one per FX pair | Holds the side with a positive broker swap (`Symbol.SwapLong/Short`), vol-targeted; flat when US500 < 200d SMA or its vol is in the top 20% (risk-off). |
| `IndexRegimeSwing` | Daily, one per index | Regime long (bull & calm) + RSI(2) pullback + turn-of-month, vol-targeted, with rebalance buffer. |
| `SessionBreakout` | H1/M15, FX & gold | EXPERIMENTAL: Asian-range breakout, OCO stop orders, risk % per trade, flat at 20:00 UTC. Backtest/demo only. |

Install: double-click the `.algo` file (or create a new cBot in cTrader Automate and paste
the `.cs`). All bots have a drawdown kill-switch and size from `Symbol.TickValue`, so
they work with £-per-point spread-bet accounts. To run unattended, use cTrader Cloud
(“Run in cloud”) or a small Windows VPS.

**Risk warning:** leveraged spread betting can lose more than a small deposit quickly.
Nothing here is financial advice; backtests are not forecasts.
