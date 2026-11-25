# expect: LA07
# Signal from today's close vs its SMA is paid the same day's open-to-close return.
import numpy as np
import pandas as pd

df = pd.read_csv("xlk.csv", index_col="date", parse_dates=True)
df["sma"] = df["close"].rolling(50).mean()
df["signal"] = (df["close"] > df["sma"]).astype(int)
df["intraday"] = df["close"] / df["open"] - 1
df["strat"] = df["signal"] * df["intraday"] - df["signal"] * 0.0005

ann = df["strat"].mean() * 252
vol = df["strat"].std() * np.sqrt(252)
print(f"Return {ann:.2%}  Vol {vol:.2%}  Sharpe {ann / vol:.2f}")
