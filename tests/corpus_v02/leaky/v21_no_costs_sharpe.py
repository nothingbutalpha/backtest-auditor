# expect: CO01:possible
import numpy as np
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["fast"] = df["close"].rolling(10).mean()
df["slow"] = df["close"].rolling(50).mean()
df["position"] = (df["fast"] > df["slow"]).astype(int).shift(1)
df["strategy"] = df["position"] * df["close"].pct_change()
sharpe = df["strategy"].mean() / df["strategy"].std() * np.sqrt(252)
print(f"Sharpe {sharpe:.2f}, total {(1 + df['strategy']).prod() - 1:.1%}")
