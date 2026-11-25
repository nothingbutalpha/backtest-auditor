# expect: LA13:high
import numpy as np
import pandas as pd

vol = pd.read_csv("volume.csv")["volume"].to_numpy()
events = np.flatnonzero(vol > np.nanmedian(vol) * 3)
signal = np.zeros(len(vol))
for s in events:
    if s + 6 >= len(vol):
        continue
    post = vol[s + 1: s + 6].mean()
    if post > vol[s]:
        signal[s] = 1
print(signal.sum())
