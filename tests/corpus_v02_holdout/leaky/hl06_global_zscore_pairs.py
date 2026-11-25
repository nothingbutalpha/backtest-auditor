# expect: LA09
# Pairs z-score uses the full-sample mean and std of the spread.
import numpy as np
import pandas as pd

df = pd.read_csv("ko_pep.csv", index_col="date", parse_dates=True)
spread = np.log(df["KO"]) - np.log(df["PEP"])
z = (spread - spread.mean()) / spread.std()

pos = pd.Series(np.nan, index=df.index)
pos[z > 2] = -1
pos[z < -2] = 1
pos[z.abs() < 0.5] = 0
pos = pos.ffill().fillna(0).shift(1)

spread_ret = spread.diff()
pnl = pos * spread_ret - pos.diff().abs().fillna(0) * 0.001
print("Sharpe", pnl.mean() / pnl.std() * np.sqrt(252))
