"""Statistical validation: block-bootstrap Sharpe CI and Deflated Sharpe Ratio
(Bailey & Lopez de Prado 2014) accounting for the number of variants tried."""
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis


def block_bootstrap_sharpe(r, n_boot=2000, block=21, seed=0):
    rng = np.random.default_rng(seed)
    x = r.dropna().values
    n = len(x)
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, n // block + 1)
        s = np.concatenate([x[i:i + block] for i in starts])[:n]
        out.append(s.mean() / s.std() * np.sqrt(252))
    return np.percentile(out, [2.5, 50, 97.5]), (np.array(out) > 0).mean()


def deflated_sharpe(r, n_trials, trial_sharpe_std_annual):
    """Probability that the true Sharpe > the best Sharpe expected by luck from
    n_trials independent tries."""
    x = r.dropna().values
    T = len(x)
    sr = x.mean() / x.std()                         # per-period
    v = (trial_sharpe_std_annual / np.sqrt(252)) ** 2
    g = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * np.e)))
    sk, ku = skew(x), kurtosis(x, fisher=False)
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return norm.cdf(z), sr0 * np.sqrt(252)


if __name__ == "__main__":
    port = pd.read_csv("results/ensemble_daily.csv", index_col=0, parse_dates=True).iloc[:, 0]
    trials = pd.read_csv("results/daily_grid.csv")
    for lbl, r in [("full 2004-2026", port), ("OOS 2018-2026", port.loc["2018":])]:
        ci, p = block_bootstrap_sharpe(r)
        print(f"{lbl}: Sharpe 95% CI [{ci[0]:.2f}, {ci[2]:.2f}]  P(Sharpe>0)={p:.0%}")
    # ~60 configurations were tried across the research; dispersion of their Sharpes ~0.3
    for n in (20, 60):
        dsr, sr0 = deflated_sharpe(port, n, 0.30)
        print(f"Deflated Sharpe (N={n} trials): P(real edge)={dsr:.0%}  luck hurdle={sr0:.2f}")
