# expect: LA10
# Twin of clean/b02: the IC of date t (which needs returns up to t+5) scales the score at t.
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
c = df.groupby("ticker")["close"]
df["raw"] = -c.pct_change(5)
df["fwd_ret"] = c.shift(-5) / df["close"] - 1
daily_ic = df.groupby("date").apply(lambda d: d["raw"].corr(d["fwd_ret"], method="spearman"))
df["ic_today"] = df["date"].map(daily_ic)
df["signal"] = df["raw"] * df["ic_today"]
