# expect: none
# The execution model: decide at the close of t, fill at the next opens.
import pandas as pd

df = pd.read_csv("spy.csv", parse_dates=["date"], index_col="date")
sig = (df["close"] > df["close"].rolling(50).mean()).astype(int)
entry_px = df["open"].shift(-1)
exit_px = df["open"].shift(-2)
costs = 0.0005 * sig.diff().abs()
trade = sig * (exit_px / entry_px - 1) - costs
print(trade.sum())
