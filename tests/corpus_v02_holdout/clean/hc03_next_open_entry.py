# expect: none
# Looks like a same-bar fill, but the close-based signal enters at the next open and exits the open after.
import numpy as np
import pandas as pd

df = pd.read_csv("dia.csv", index_col="date", parse_dates=True)
mom = df["close"] / df["close"].shift(5) - 1
df["sig"] = np.sign(mom).fillna(0)

entry = df["open"].shift(-1)
exit_ = df["open"].shift(-2)
df["trade_ret"] = exit_ / entry - 1
df["net"] = df["sig"] * df["trade_ret"] - 0.0005 * df["sig"].diff().abs().fillna(0)

r = df["net"].dropna()
print("Sharpe", r.mean() / r.std() * np.sqrt(252))
