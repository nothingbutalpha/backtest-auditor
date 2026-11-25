import yfinance as yf
import numpy as np
df = yf.download("MSFT", period="10y")
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
df["RSI"] = 100 - 100 / (1 + gain / loss)
df["Position"] = np.where(df["RSI"] < 30, 1, np.where(df["RSI"] > 70, 0, np.nan))
df["Position"] = df["Position"].ffill().fillna(0)
df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
df["Strategy"] = df["Position"] * df["Returns"]
print(np.exp(df["Strategy"].sum()))
