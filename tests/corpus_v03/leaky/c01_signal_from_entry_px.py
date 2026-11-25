# expect: LA01|LA10:high
# Twin of clean/c01: the signal compares tomorrow's open with today's close (gap known in advance).
import pandas as pd

df = pd.read_csv("spy.csv", parse_dates=["date"], index_col="date")
entry_px = df["open"].shift(-1)
exit_px = df["open"].shift(-2)
df["trade_ret"] = exit_px / entry_px - 1
df["signal"] = (entry_px > df["close"] * 1.002).astype(int)
df["strategy"] = df["signal"] * df["trade_ret"] - 0.0005 * df["signal"].diff().abs()
