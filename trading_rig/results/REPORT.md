# Research agent report - 2026-11-25

Variants examined to date (for deflated Sharpe): 60

## Strategy gates

| strategy         |   IS_sharpe |   OOS_sharpe | OOS_CI        | OOS_cagr   | OOS_maxdd   | DSR   | G1   | G2   | G3   | G4   | verdict   |
|:-----------------|------------:|-------------:|:--------------|:-----------|:------------|:------|:-----|:-----|:-----|:-----|:----------|
| A_index_pullback |        0.59 |         0.14 | [-0.48, 0.89] | 1.0%       | -42.5%      | 8%    | fail | PASS | PASS | fail | REJECT    |
| B_turn_of_month  |        0.19 |         0.16 | [-0.65, 0.97] | 1.2%       | -35.7%      | 1%    | fail | fail | PASS | fail | REJECT    |
| C_fx_carry       |        0.18 |         0.36 | [-0.23, 1.05] | 4.0%       | -25.5%      | 2%    | PASS | fail | PASS | fail | WATCHLIST |
| D_index_regime   |        0.55 |         0.12 | [-0.58, 0.79] | 0.7%       | -29.9%      | 6%    | fail | PASS | PASS | fail | REJECT    |
| ENSEMBLE         |        0.57 |         0.3  | [-0.41, 1.00] | 3.0%       | -36.2%      | 12%   | PASS | PASS | PASS | fail | WATCHLIST |

## Drift monitor (last 6 months vs own history)

| sleeve           | last_6m   | percentile   | flag   |
|:-----------------|:----------|:-------------|:-------|
| A_index_pullback | -2.2%     | 26%          | ok     |
| B_turn_of_month  | 7.4%      | 73%          | ok     |
| C_fx_carry       | -2.1%     | 32%          | ok     |
| D_index_regime   | -8.1%     | 8%           | ok     |
