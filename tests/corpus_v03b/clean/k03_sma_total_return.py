# expect: none
# CO01 class: a simple lagged SMA rule with a total return and no cost model. Costs are a realism caveat
# (CO01 possible), not a lookahead red flag.
import pandas as pd

df = pd.read_csv("spy.csv", parse_dates=["date"], index_col="date")
df["position"] = (df["close"] > df["close"].rolling(200).mean()).astype(int).shift(1)
df["strategy"] = df["position"] * df["close"].pct_change()
print("total return", (1 + df["strategy"].fillna(0)).prod() - 1)
