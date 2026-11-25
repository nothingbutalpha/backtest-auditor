# expect: CO01:possible
# Lagged MA-crossover returns with Sharpe and CAGR reported, no commissions or slippage.
import numpy as np
import pandas as pd

df = pd.read_csv("iwm.csv", index_col="date", parse_dates=True)
df["ret"] = df["close"].pct_change()
df["fast"] = df["close"].rolling(20).mean()
df["slow"] = df["close"].rolling(100).mean()
df["pos"] = (df["fast"] > df["slow"]).astype(float).shift(1)
df["strat"] = df["pos"] * df["ret"]
sharpe = df["strat"].mean() / df["strat"].std() * np.sqrt(252)
cagr = (1 + df["strat"].fillna(0)).prod() ** (252 / len(df)) - 1
print(f"Sharpe {sharpe:.2f}  CAGR {cagr:.1%}")
