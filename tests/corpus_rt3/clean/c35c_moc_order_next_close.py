# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["signal"] = (df["close"] > df["close"].rolling(20).mean()).astype(int)
df["fwd_ret"] = df["close"].pct_change().shift(-1)
df["strat"] = df["signal"] * df["fwd_ret"] - 0.0005 * df["signal"].diff().abs()
print("sharpe", df["strat"].mean() / df["strat"].std() * 252 ** 0.5)
