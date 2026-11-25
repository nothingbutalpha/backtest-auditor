# expect: none
# Class 3: a bar loop that fills at a precomputed next-open array.
import numpy as np
import pandas as pd

df = pd.read_csv("es.csv", parse_dates=["date"], index_col="date")
sig = (df["close"] > df["close"].rolling(20).mean()).to_numpy()
entry_px = df["open"].shift(-1).to_numpy()
exit_px = df["open"].shift(-2).to_numpy()
pnl = np.zeros(len(df))
for i in range(len(df) - 2):
    if sig[i]:
        pnl[i] = exit_px[i] / entry_px[i] - 1 - 0.0004
print(pnl.sum())
