# expect: LA07:high, CO01:possible, DQ01:info
# The kind of strategy a chatbot writes when asked for "a simple moving average crossover backtest".
import numpy as np
import yfinance as yf

df = yf.download("SPY", start="2010-01-01", end="2024-12-31")
df["SMA20"] = df["Close"].rolling(window=20).mean()
df["SMA50"] = df["Close"].rolling(window=50).mean()
df["Position"] = np.where(df["SMA20"] > df["SMA50"], 1, 0)
df["Returns"] = df["Close"].pct_change()
df["Strategy"] = df["Position"] * df["Returns"]
cumulative = (1 + df["Strategy"]).cumprod()
print("Total return:", cumulative.iloc[-1] - 1)
