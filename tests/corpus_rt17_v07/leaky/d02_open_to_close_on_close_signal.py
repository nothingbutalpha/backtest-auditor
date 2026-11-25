# leak: 6
import pandas as pd
k = pd.read_csv("nq_daily.csv", index_col=0, parse_dates=True)
up = k["close"] > k["close"].rolling(10).mean()
intraday = k["close"] / k["open"] - 1
gain = intraday.where(up, 0.0)
print(gain.add(1).prod())
