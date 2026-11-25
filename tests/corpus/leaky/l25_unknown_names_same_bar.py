# expect: LA07:high, CO01:possible
# From the red-team review of v0.1 (regression case).
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["go"] = (df["close"] > df["close"].rolling(20).mean()).astype(int)
df["gain"] = df["go"] * df["close"].pct_change()
print((1 + df["gain"]).prod())
