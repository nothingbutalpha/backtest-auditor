# expect: LA10:high
import numpy as np
import pandas as pd
px = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["close"].to_numpy()
tomorrow = np.roll(px, -1)
signal = np.where(tomorrow > px, 1, 0)
ret = np.r_[0.0, px[1:] / px[:-1] - 1]
pnl = signal[:-1] * ret[1:] - 0.0005
print(pnl.mean() / pnl.std() * np.sqrt(252))
