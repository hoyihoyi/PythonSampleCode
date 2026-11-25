"""Strategy library. Each function: (df, **params) -> forecast in [-1, 1]
(decided at bar close). Vol targeting is applied by the caller.
"""
import numpy as np
import pandas as pd


# ---------------- indicators ----------------

def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def atr(df, n=20):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(c, n=2):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def efficiency_ratio(c, n=20):
    """Kaufman ER: |net move| / sum |moves|. ~1 = clean trend, ~0 = chop."""
    net = (c - c.shift(n)).abs()
    path = c.diff().abs().rolling(n).sum()
    return (net / path).fillna(0)


def price_vol(c, span=36):
    """daily price-unit volatility"""
    return c.diff().ewm(span=span, min_periods=span).std()


# ---------------- trend / momentum ----------------

def ewmac(df, pairs=((8, 32), (16, 64), (32, 128), (64, 256)), cap=2.0):
    """Multi-speed EMA crossover, normalised by price vol, averaged.
    Continuous forecast -> naturally sizes up in strong trends."""
    c = df["Close"]
    pv = price_vol(c)
    fs = []
    for f, s in pairs:
        raw = (ema(c, f) - ema(c, s)) / pv
        raw = raw / raw.abs().expanding(min_periods=250).mean()  # scale to avg |1|
        fs.append(raw.clip(-cap, cap))
    return (pd.concat(fs, axis=1).mean(axis=1) / cap).fillna(0)


def tsmom(df, lookbacks=(21, 63, 126, 252)):
    """Time-series momentum: average sign of past returns over several horizons."""
    c = df["Close"]
    s = sum(np.sign(c / c.shift(L) - 1) for L in lookbacks) / len(lookbacks)
    return s.fillna(0)


def donchian(df, entry=55, exit=20):
    """Turtle-style breakout. Long on N-day high, exit on M-day low; mirror for shorts."""
    c, h, l = df["Close"], df["High"], df["Low"]
    hi_e, lo_e = h.rolling(entry).max().shift(1), l.rolling(entry).min().shift(1)
    hi_x, lo_x = h.rolling(exit).max().shift(1), l.rolling(exit).min().shift(1)
    pos = np.zeros(len(c))
    cv, he, le, hx, lx = (x.values for x in (c, hi_e, lo_e, hi_x, lo_x))
    for i in range(1, len(c)):
        p = pos[i - 1]
        if p == 0:
            if cv[i] > he[i]:
                p = 1
            elif cv[i] < le[i]:
                p = -1
        elif p == 1 and cv[i] < lx[i]:
            p = -1 if cv[i] < le[i] else 0
        elif p == -1 and cv[i] > hx[i]:
            p = 1 if cv[i] > he[i] else 0
        pos[i] = p
    return pd.Series(pos, index=c.index)


# ---------------- mean reversion ----------------

def rsi2_mr(df, lo=10, hi=90, trend_n=200, exit_n=5, allow_short=False):
    """Connors-style short-term pullback in the direction of the long trend."""
    c = df["Close"]
    r = rsi(c, 2)
    up = c > c.rolling(trend_n).mean()
    sma_x = c.rolling(exit_n).mean()
    pos = np.zeros(len(c))
    cv, rv, uv, xv = c.values, r.values, up.values, sma_x.values
    for i in range(1, len(c)):
        p = pos[i - 1]
        if p == 0:
            if uv[i] and rv[i] < lo:
                p = 1
            elif allow_short and (not uv[i]) and rv[i] > hi:
                p = -1
        elif p == 1 and cv[i] > xv[i]:
            p = 0
        elif p == -1 and cv[i] < xv[i]:
            p = 0
        pos[i] = p
    return pd.Series(pos, index=c.index)


def ibs_mr(df, lo=0.2, trend_n=200):
    """Internal bar strength: close near the day's low -> next day bounce (index long only)."""
    c = df["Close"]
    ibs = (c - df["Low"]) / (df["High"] - df["Low"]).replace(0, np.nan)
    up = c > c.rolling(trend_n).mean()
    return ((ibs < lo) & up).astype(float)


def zscore_mr(df, n=20, entry=2.0, exit=0.5):
    """Fade stretched moves away from the n-bar mean (symmetric)."""
    c = df["Close"]
    z = (c - c.rolling(n).mean()) / c.rolling(n).std()
    pos = np.zeros(len(c))
    zv = z.values
    for i in range(1, len(c)):
        p = pos[i - 1]
        if p == 0:
            if zv[i] < -entry:
                p = 1
            elif zv[i] > entry:
                p = -1
        elif p == 1 and zv[i] > -exit:
            p = 0
        elif p == -1 and zv[i] < exit:
            p = 0
        pos[i] = p
    return pd.Series(pos, index=c.index)


# ---------------- regime-adaptive ----------------

def regime(df, er_n=60, er_lookback=500):
    """Trend-regime score in [0,1]: percentile of the efficiency ratio vs its own history."""
    er = efficiency_ratio(df["Close"], er_n)
    return er.rolling(er_lookback, min_periods=120).rank(pct=True).fillna(0.5)


def adaptive(df, er_n=60, hi=0.6, lo=0.4, mr_weight=1.0):
    """Regime switch: trend-following (ewmac) when the market is trending,
    short-term mean reversion (zscore, plus RSI2 pullbacks) when it is choppy,
    blend in between."""
    reg = regime(df, er_n)
    trend = ewmac(df)
    mr = 0.5 * zscore_mr(df, 10, 1.5, 0.0) + 0.5 * rsi2_mr(df, allow_short=True)
    w = ((reg - lo) / (hi - lo)).clip(0, 1)
    return (w * trend + (1 - w) * mr_weight * mr).clip(-1, 1)


def trend_regime_filtered(df, er_n=60, thresh=0.3):
    """Trend-following only when the regime score says trends are present; flat otherwise."""
    reg = regime(df, er_n)
    return ewmac(df) * (reg > thresh)


DAILY = {
    "ewmac": ewmac,
    "tsmom": tsmom,
    "donchian": donchian,
    "rsi2_mr": rsi2_mr,
    "ibs_mr": ibs_mr,
    "zscore_mr": zscore_mr,
    "adaptive": adaptive,
    "trend_regime": trend_regime_filtered,
}


# ---------------- equity-index long-bias regime family ----------------

def buy_hold(df):
    return pd.Series(1.0, index=df.index)


def trend_long(df, n=200):
    """Long only while price is above its n-day average (regime = bull)."""
    c = df["Close"]
    return (c > c.rolling(n).mean()).astype(float)


def vol_regime_long(df, n=200, vol_n=20, vol_pct=0.8, lookback=750):
    """Long in bull regime unless short-term vol is in the top (1-vol_pct)
    of its history (crash regime) -> flat."""
    c = df["Close"]
    rv = c.pct_change().rolling(vol_n).std()
    hi_vol = rv.rolling(lookback, min_periods=250).rank(pct=True) > vol_pct
    return ((c > c.rolling(n).mean()) & ~hi_vol).astype(float)


def bull_dip(df, n=200, lo=10, w_base=0.5):
    """Regime-adaptive index model:
       bull regime  -> base long (w_base) + full size on RSI2 pullbacks
       bear regime  -> flat, except deep-oversold RSI2 bounces at half size."""
    c = df["Close"]
    bull = c > c.rolling(n).mean()
    pull = rsi2_mr(df, lo=lo, trend_n=n)                 # long pullbacks in bull
    r2 = rsi(c, 2)
    bear_bounce = rsi2_mr(df.assign(Close=c), lo=5, trend_n=1)  # trend filter off
    bear_bounce = bear_bounce * (~bull)
    f = bull * w_base + pull * (1 - w_base) + 0.5 * bear_bounce
    return f.clip(0, 1)


DAILY.update(buy_hold=buy_hold, trend_long=trend_long, vol_regime_long=vol_regime_long,
             bull_dip=bull_dip)


# ---------------- calendar / microstructure anomalies ----------------

def turn_of_month(df, before=1, after=3):
    """Long from the close `before+1` trading days before month end until the
    close of trading day `after` of the new month (flows: pensions/payroll)."""
    idx = df.index
    m = pd.Series(idx.to_period("M"), index=idx)
    day_from_end = m.groupby(m).cumcount(ascending=False)   # 0 = last day of month
    day_of_month = m.groupby(m).cumcount()                  # 0 = first day
    # position decided at close t is held during t+1
    hold_next = (day_from_end.shift(-1) <= before - 1) | (day_of_month.shift(-1) <= after - 1)
    hold_next = hold_next | (day_from_end == before)  # enter at close `before` days pre-month-end
    return hold_next.fillna(False).astype(float)


DAILY.update(turn_of_month=turn_of_month)


# ---------------- FX carry with risk regime ----------------
_RISK = None


def risk_off(index_name="US500", vol_n=20, pct=0.8, lookback=750, trend_n=200):
    """Global risk regime from equities: risk-off if US500 is below its 200d
    average AND/OR short-term vol is in its top 20% (carry-crash regime)."""
    global _RISK
    if _RISK is None:
        from data import load
        c = load(index_name, "1d")["Close"]
        rv = c.pct_change().rolling(vol_n).std()
        hi = rv.rolling(lookback, min_periods=250).rank(pct=True) > pct
        _RISK = (hi | (c < c.rolling(trend_n).mean())).astype(float)
    return _RISK


def fx_carry_sig(df, name, scale=0.02, regime=True):
    from engine import fx_carry
    d = fx_carry(name).reindex(df.index, method="ffill").fillna(0)
    f = (d / scale).clip(-1, 1)
    if regime:
        ro = risk_off().reindex(df.index, method="ffill").fillna(0)
        f = f * (1 - ro)
    return f


# ---------------- relative value: index pairs ----------------

def pair_signal(a: pd.DataFrame, b: pd.DataFrame, n=20, entry=2.0, exit=0.0, beta_n=120):
    """Mean reversion of the vol-adjusted log spread between two correlated
    indices. Returns (pos_a, pos_b) in units of 'one side' (vol targeting by caller).
    Long the laggard / short the leader when the spread is stretched."""
    la, lb = np.log(a["Close"]), np.log(b["Close"])
    ra, rb = la.diff(), lb.diff()
    beta = (ra.rolling(beta_n).cov(rb) / rb.rolling(beta_n).var()).shift(1)
    spread = (ra - beta * rb).cumsum()
    z = (spread - spread.rolling(n).mean()) / spread.rolling(n).std()
    pos = np.zeros(len(z))
    zv = z.values
    for i in range(1, len(zv)):
        p = pos[i - 1]
        if p == 0:
            p = -1 if zv[i] > entry else (1 if zv[i] < -entry else 0)
        elif (p == 1 and zv[i] > -exit) or (p == -1 and zv[i] < exit):
            p = 0
        pos[i] = p
    s = pd.Series(pos, index=z.index)
    return s, -s * beta.fillna(1)


# ---------------- statistical regime model (walk-forward Markov switching) ----------------

def markov_regime_prob(c: pd.Series, refit_every="YS", min_train=520, k=2):
    """Probability of the calm/bull regime from a 2-state Markov-switching model
    on weekly returns with switching variance. Refit at the start of every year
    on all data to that point (expanding window); filtered (not smoothed)
    probabilities only -> no look-ahead."""
    import statsmodels.api as sm
    w = np.log(c).diff().resample("W-FRI").sum().dropna() * 100
    out = pd.Series(np.nan, index=w.index)
    starts = w.resample(refit_every).first().index
    for s in starts:
        train = w.loc[:s - pd.Timedelta(days=1)]
        if len(train) < min_train:
            continue
        nxt = w.loc[s:s + pd.offsets.YearEnd(0)]
        try:
            mod = sm.tsa.MarkovRegression(train, k_regimes=k, trend="c", switching_variance=True)
            fit = mod.fit(disp=False)
            calm = int(np.argmin(fit.params[[f"sigma2[{i}]" for i in range(k)]].values))
            full = w.loc[:nxt.index[-1]] if len(nxt) else train
            # filter forward with fixed params (no re-estimation inside the year)
            res = sm.tsa.MarkovRegression(full, k_regimes=k, trend="c",
                                          switching_variance=True).filter(fit.params)
            p = res.filtered_marginal_probabilities[calm]
            out.loc[nxt.index] = p.loc[nxt.index].values
        except Exception:
            continue
    # weekly prob known at Friday close -> apply from next trading day
    return out


def hmm_regime_long(df, thresh=0.5):
    p = markov_regime_prob(df["Close"])
    daily = p.reindex(df.index, method="ffill")
    return (daily > thresh).astype(float).fillna(0)


DAILY.update(hmm_regime_long=hmm_regime_long)


def rsi2_mr_guarded(df, lo=10, trend_n=200, exit_n=5, shock=1.5, stop_atr=3.0, max_hold=10):
    """Regime-guarded pullback: as rsi2_mr, but
       * no new entries in a volatility-shock regime (10d realised vol > shock x 100d vol)
       * protective stop: exit if close < entry - stop_atr * ATR(20)
       * time stop: exit after max_hold bars."""
    c = df["Close"]
    r = rsi(c, 2)
    up = c > c.rolling(trend_n).mean()
    ret = c.pct_change()
    vshock = ret.rolling(10).std() > shock * ret.rolling(100).std()
    sma_x = c.rolling(exit_n).mean()
    a = atr(df, 20)
    pos = np.zeros(len(c))
    cv, rv, uv, xv, sv, av = c.values, r.values, up.values, sma_x.values, vshock.values, a.values
    entry, held = 0.0, 0
    for i in range(1, len(c)):
        p = pos[i - 1]
        if p == 0:
            if uv[i] and rv[i] < lo and not sv[i]:
                p, entry, held = 1, cv[i], 0
        else:
            held += 1
            if cv[i] > xv[i] or cv[i] < entry - stop_atr * av[i] or held >= max_hold:
                p = 0
        pos[i] = p
    return pd.Series(pos, index=c.index)
