# expect: none
# Class 2: information coefficient per date between a lagged score and the forward return.
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
c = df.groupby("ticker")["close"]
df["score"] = -c.pct_change(5)
df["fwd_ret"] = c.shift(-5) / df["close"] - 1
daily_ic = df.groupby("date").apply(lambda d: d["score"].corr(d["fwd_ret"], method="spearman"))
print("mean IC", daily_ic.mean(), "t", daily_ic.mean() / daily_ic.std() * len(daily_ic) ** 0.5)
