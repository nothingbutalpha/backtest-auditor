# expect: LA07:medium@7
import numpy as np
import pandas as pd

df = pd.read_csv("xlf.csv", index_col=0, parse_dates=True)
sig = df["close"] < df["close"].rolling(20).min().shift(1)
df["fill_px"] = np.where(sig.shift(1, fill_value=False), df["low"], np.nan)
df["exit_px"] = df["close"].shift(-3)
t = df.dropna(subset=["fill_px", "exit_px"])
print((t["exit_px"] / t["fill_px"] - 1).mean())
