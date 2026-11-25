import yfinance as yf
import numpy as np
df = yf.download("QQQ", start="2010-01-01")
df["MA20"] = df["Close"].rolling(20).mean()
df["STD20"] = df["Close"].rolling(20).std()
df["Upper"] = df["MA20"] + 2 * df["STD20"]
df["Lower"] = df["MA20"] - 2 * df["STD20"]
df["Position"] = 0
df.loc[df["Close"] < df["Lower"], "Position"] = 1
df.loc[df["Close"] > df["Upper"], "Position"] = -1
df["Market_Return"] = df["Close"].pct_change()
df["Strategy_Return"] = df["Market_Return"] * df["Position"]
print((1 + df["Strategy_Return"]).prod())
