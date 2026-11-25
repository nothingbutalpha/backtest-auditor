import yfinance as yf
import pandas as pd
data = yf.download("AAPL", start="2015-01-01", end="2024-01-01")
data["SMA50"] = data["Close"].rolling(window=50).mean()
data["SMA200"] = data["Close"].rolling(window=200).mean()
data["Signal"] = 0
data.loc[data["SMA50"] > data["SMA200"], "Signal"] = 1
data["Daily_Return"] = data["Close"].pct_change()
data["Strategy_Return"] = data["Signal"] * data["Daily_Return"]
cumulative = (1 + data["Strategy_Return"]).cumprod()
print(cumulative.iloc[-1])
