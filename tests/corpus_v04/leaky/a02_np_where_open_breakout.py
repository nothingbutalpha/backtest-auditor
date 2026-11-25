# expect: LA07:high
import numpy as np
import pandas as pd

px = pd.read_parquet("qqq.parquet")
hi20 = px["High"].rolling(20).max().shift(1)
signal = px["Close"] > hi20
buy_price = np.where(signal, px["Open"], np.nan)
sell_price = px["Close"].shift(-10).to_numpy()
r = sell_price / buy_price - 1 - 0.0005
print(np.nanmean(r), np.isfinite(r).sum())
