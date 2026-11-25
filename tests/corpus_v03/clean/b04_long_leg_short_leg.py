# expect: none
# Class 2: the legs of a decile portfolio are the forward holding-period return of the selected names.
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
px = df.groupby("ticker")["close"]
df["mom"] = px.pct_change(126).groupby(df["ticker"]).shift(1)
df["fwd_ret"] = px.shift(-21) / df["close"] - 1
df["decile"] = df.groupby("date")["mom"].transform(lambda s: pd.qcut(s, 10, labels=False, duplicates="drop"))
long_leg = df[df["decile"] == 9].groupby("date")["fwd_ret"].mean()
short_leg = df[df["decile"] == 0].groupby("date")["fwd_ret"].mean()
spread = long_leg - short_leg - 0.002
print(spread.describe())
