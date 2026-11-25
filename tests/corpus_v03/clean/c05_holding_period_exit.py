# expect: none
# Class 3: enter at the next open, hold HOLD days, exit at the open.
import pandas as pd

HOLD = 5
df = pd.read_csv("spy.csv", parse_dates=["date"], index_col="date")
df["rsi_proxy"] = df["close"].pct_change().rolling(14).mean()
df["signal"] = (df["rsi_proxy"] < -0.003).astype(int)
fill_px = df["open"].shift(-1)
exit_px = df["open"].shift(-(HOLD + 1))
df["hold_ret"] = exit_px / fill_px - 1
trades = df.loc[df["signal"] == 1, "hold_ret"] - 0.001
print(trades.mean(), (trades > 0).mean(), len(trades))
