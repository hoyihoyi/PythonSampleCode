"""Instrument universe mapped to Pepperstone spread-betting markets, with a
conservative cost model.

Costs are deliberately pessimistic so that a strategy that survives here has
margin for error live:
  spread      - full bid/ask spread in price units (typical Pepperstone SB, core hours)
  slip        - extra adverse slippage per side, in price units
  markup      - financing markup over/under the benchmark rate (annual).
                Long pays benchmark + markup, short receives benchmark - markup
                (Pepperstone SB style). Benchmark = historical US 3m T-bill (FRED).
  div         - dividend yield credited to longs / debited to shorts for
                price-return indices (0 for total-return GER40 and non-equities)
  FX          - for FX the carry differential is ignored; both sides pay `markup`.
  (fin_long / fin_short are legacy flat rates, used only if benchmark missing)
  intraday_spread_mult - how much wider spreads are outside core hours (hourly tests)
"""

INSTRUMENTS = {
    # name      yahoo      class     spread    slip     fin_long fin_short
    "US500":   dict(yf="^GSPC",   cls="index", spread=0.6,     slip=0.3,    fin_long=0.065, fin_short=0.010, markup=0.025, div=0.014),
    "NAS100":  dict(yf="^NDX",    cls="index", spread=1.5,     slip=0.75,   fin_long=0.065, fin_short=0.010, markup=0.025, div=0.008),
    "US30":    dict(yf="^DJI",    cls="index", spread=2.4,     slip=1.2,    fin_long=0.065, fin_short=0.010, markup=0.025, div=0.019),
    "GER40":   dict(yf="^GDAXI",  cls="index", spread=1.2,     slip=0.6,    fin_long=0.045, fin_short=0.010, markup=0.025, div=0.0),
    "UK100":   dict(yf="^FTSE",   cls="index", spread=1.0,     slip=0.5,    fin_long=0.065, fin_short=0.010, markup=0.025, div=0.035),
    "JPN225":  dict(yf="^N225",   cls="index", spread=8.0,     slip=4.0,    fin_long=0.030, fin_short=0.020, markup=0.025, div=0.018),
    "XAUUSD":  dict(yf="GC=F",    cls="metal", spread=0.30,    slip=0.15,   fin_long=0.065, fin_short=0.010, markup=0.025, div=0),
    "XAGUSD":  dict(yf="SI=F",    cls="metal", spread=0.025,   slip=0.012,  fin_long=0.065, fin_short=0.010, markup=0.025, div=0),
    "WTI":     dict(yf="CL=F",    cls="energy",spread=0.03,    slip=0.015,  fin_long=0.040, fin_short=0.040, markup=0.0, div=0),
    "EURUSD":  dict(yf="EURUSD=X",cls="fx",    spread=0.00008, slip=0.00004,fin_long=0.015, fin_short=0.015, markup=0.01, div=0),
    "GBPUSD":  dict(yf="GBPUSD=X",cls="fx",    spread=0.00012, slip=0.00006,fin_long=0.015, fin_short=0.015, markup=0.01, div=0),
    "USDJPY":  dict(yf="JPY=X",   cls="fx",    spread=0.010,   slip=0.005,  fin_long=0.015, fin_short=0.015, markup=0.01, div=0),
    "AUDUSD":  dict(yf="AUDUSD=X",cls="fx",    spread=0.00010, slip=0.00005,fin_long=0.015, fin_short=0.015, markup=0.01, div=0),
    "USDCAD":  dict(yf="CAD=X",   cls="fx",    spread=0.00015, slip=0.00008,fin_long=0.015, fin_short=0.015, markup=0.01, div=0),
    "BTCUSD":  dict(yf="BTC-USD", cls="crypto",spread=40.0,    slip=20.0,   fin_long=0.20,  fin_short=0.20, markup=0.15, div=0),
}

# ---- extended universe (added for diversification of trend strategies) ----
_FX = dict(cls="fx", markup=0.01, div=0, fin_long=0.015, fin_short=0.015)
_IX = dict(cls="index", markup=0.025, fin_long=0.065, fin_short=0.01)
INSTRUMENTS.update({
    "EURJPY": dict(yf="EURJPY=X", spread=0.015, slip=0.008, **_FX),
    "GBPJPY": dict(yf="GBPJPY=X", spread=0.025, slip=0.012, **_FX),
    "EURGBP": dict(yf="EURGBP=X", spread=0.00012, slip=0.00006, **_FX),
    "AUDJPY": dict(yf="AUDJPY=X", spread=0.015, slip=0.008, **_FX),
    "NZDUSD": dict(yf="NZDUSD=X", spread=0.00015, slip=0.00008, **_FX),
    "USDCHF": dict(yf="CHF=X", spread=0.00012, slip=0.00006, **_FX),
    "EURCHF": dict(yf="EURCHF=X", spread=0.00015, slip=0.00008, **_FX),
    "HK50":   dict(yf="^HSI", spread=8.0, slip=4.0, div=0.035, **_IX),
    "AUS200": dict(yf="^AXJO", spread=1.0, slip=0.5, div=0.04, **_IX),
    "EU50":   dict(yf="^STOXX50E", spread=1.2, slip=0.6, div=0.03, **_IX),
    "FRA40":  dict(yf="^FCHI", spread=1.0, slip=0.5, div=0.03, **_IX),
    "US2000": dict(yf="^RUT", spread=0.3, slip=0.15, div=0.012, **_IX),
    "XPTUSD": dict(yf="PL=F", cls="metal", spread=2.0, slip=1.0, markup=0.025, div=0, fin_long=0.065, fin_short=0.01),
    "ETHUSD": dict(yf="ETH-USD", cls="crypto", spread=3.0, slip=1.5, markup=0.15, div=0, fin_long=0.2, fin_short=0.2),
})

INTRADAY_SPREAD_MULT = 1.5  # hourly bars include out-of-hours trading -> wider spreads
