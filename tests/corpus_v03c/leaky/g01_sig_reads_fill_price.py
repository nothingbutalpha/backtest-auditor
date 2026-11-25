# expect: LA01|LA10:high
# Twin of clean/g01: the signal reads the next open (a gap known in advance).
import pandas as pd

df = pd.read_csv("spy.csv", parse_dates=["date"], index_col="date")
entry_px = df["open"].shift(-1)
exit_px = df["open"].shift(-2)
sig = (entry_px > df["close"] * 1.002).astype(int)
costs = 0.0005 * sig.diff().abs()
trade = sig * (exit_px / entry_px - 1) - costs
