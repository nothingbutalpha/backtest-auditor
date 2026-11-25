# expect: none
# Looks like a global z-score, but the spread is normalised with a trailing 60-day rolling window.
import numpy as np
import pandas as pd

df = pd.read_csv("ko_pep.csv", index_col="date", parse_dates=True)
spread = np.log(df["KO"]) - np.log(df["PEP"])
z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()

pos = pd.Series(np.nan, index=df.index)
pos[z > 2] = -1
pos[z < -2] = 1
pos[z.abs() < 0.5] = 0
pos = pos.ffill().fillna(0).shift(1).fillna(0)

pnl = pos * spread.diff() - pos.diff().abs().fillna(0) * 0.001
print("Sharpe", pnl.mean() / pnl.std() * np.sqrt(252))
