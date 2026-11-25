# expect: LA02:high
import pandas as pd
df = pd.read_csv("prices.csv", parse_dates=["date"])
df = df.sort_values("date", ascending=False)
df["sma20"] = df["close"].rolling(20).mean()
df["signal"] = (df["close"] > df["sma20"]).astype(int)
