# expect: none
# Class 2: top and bottom quintile mean forward return through .loc masks.
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
df["fwd_ret"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
df["rank"] = df.groupby("date")["quality"].rank(pct=True)
top = df.loc[df["rank"] > 0.8].groupby("date")["fwd_ret"].mean()
bottom = df.loc[df["rank"] <= 0.2].groupby("date")["fwd_ret"].mean()
excess = top - bottom
print(excess.mean() * 252)
