# expect: none
import pandas as pd
df = pd.read_csv("prices.csv", parse_dates=["date"]).sort_values("date")
df["sma20"] = df["close"].rolling(20).mean()
df["signal"] = (df["close"] > df["sma20"]).astype(int)
latest = df.sort_values("date", ascending=False).head(10)
print(latest[["date", "close", "sma20", "signal"]])
