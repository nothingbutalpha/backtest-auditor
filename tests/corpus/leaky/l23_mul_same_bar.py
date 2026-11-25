# expect: LA07:high
import pandas as pd

df = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
df["ret"] = df["close"].pct_change()
df["signal"] = (df["close"] > df["close"].rolling(100).mean()).astype(int)
commission = 0.0005
df["strat"] = df["signal"].mul(df["ret"]) - df["signal"].diff().abs() * commission
