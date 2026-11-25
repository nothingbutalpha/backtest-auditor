# leak: 6
import pandas as pd
from scipy.ndimage import gaussian_filter1d
prices = pd.read_parquet("fx.parquet")
for col in prices:
    trend = gaussian_filter1d(prices[col].to_numpy(), sigma=4)
    weight = (prices[col] > trend).astype(int)
    pnl = weight * prices[col].pct_change().shift(-1)
    print(col, pnl.mean())
