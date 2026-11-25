# expect: LA01:high
# The signal compares tomorrow's close with today's: it reads the future.
import pandas as pd

df = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
df["signal"] = (df["close"].shift(-1) > df["close"]).astype(int)
df["ret"] = df["close"].pct_change()
fee = 0.0005
df["strategy"] = df["signal"].shift(1) * df["ret"] - df["signal"].diff().abs() * fee
print(df["strategy"].mean() / df["strategy"].std() * 252 ** 0.5)
