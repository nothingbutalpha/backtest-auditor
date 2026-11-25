import pandas as pd

df = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
df["smooth"] = df["close"].rolling(15, center=True).mean()
df["signal"] = (df["close"] > df["smooth"]).astype(int)
df.to_csv("signals.csv")
