# expect: none
import pandas as pd
df = pd.read_csv("alphavantage_daily.csv", parse_dates=["timestamp"])  # the vendor file is newest-first
df = df[::-1].reset_index(drop=True)
df["ret"] = df["close"].pct_change()
df["sma"] = df["close"].rolling(50).mean()
df["signal"] = (df["close"] > df["sma"]).astype(int)
df["strategy"] = df["signal"].shift(1) * df["ret"] - 0.0005 * df["signal"].diff().abs()
