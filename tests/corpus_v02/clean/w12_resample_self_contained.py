# expect: none
import pandas as pd
m1 = pd.read_csv("nq_1min.csv", parse_dates=["time"], index_col="time")
bars = m1["close"].resample("5min", label="right", closed="right").ohlc()
bars["ret"] = bars["close"].pct_change()
bars["signal"] = (bars["close"] > bars["close"].rolling(12).mean()).astype(int)
bars["strategy"] = bars["signal"].shift(1) * bars["ret"] - 0.0001 * bars["signal"].diff().abs()
