# expect: LA12:medium
import pandas as pd
m1 = pd.read_csv("nq_1min.csv", parse_dates=["time"], index_col="time")
hourly = m1["close"].resample("1h", closed="right").last().to_frame("h_close")
hourly["h_sma"] = hourly["h_close"].rolling(10).mean()
merged = m1.join(hourly, how="left").ffill()
merged["signal"] = (merged["close"] > merged["h_sma"]).astype(int)
