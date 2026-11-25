# expect: none
import pandas as pd

bars = pd.read_csv("xle.csv", index_col="date", parse_dates=True)
spike = (bars["volume"] > 2 * bars["volume"].rolling(20).mean().shift(1)) & (bars["close"] > bars["open"])
spike = spike.shift(1, fill_value=False)
fill_px = bars["open"][spike]
out_px = bars["close"].shift(-3)[spike]
print(((out_px / fill_px) - 1 - 0.0008).mean())
