# leak: 7
import numpy as np
import pandas as pd
k = pd.read_csv("cl_1h.csv")
lo = k["low"].to_numpy()
order_px = k["close"].to_numpy() * 0.997
hit = lo <= order_px
exit_px = np.roll(k["close"].to_numpy(), -3)
print(np.where(hit, exit_px / order_px - 1, 0.0).sum())
