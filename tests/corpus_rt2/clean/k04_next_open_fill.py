import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["sma"] = df["Close"].rolling(20).mean()
df["signal"] = (df["Close"] > df["sma"]).astype(int)
df["entry_px"] = df["Open"].shift(-1)
df["exit_px"] = df["Open"].shift(-2)
df["trade_ret"] = df["signal"] * (df["exit_px"] / df["entry_px"] - 1) - 0.001 * df["signal"]
