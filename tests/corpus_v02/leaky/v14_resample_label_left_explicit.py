# expect: LA12:medium
import pandas as pd
m1 = pd.read_csv("eurusd_1min.csv", parse_dates=["time"], index_col="time")
h4 = m1["close"].resample("4h", label="left", closed="right").last()
m1 = pd.merge_asof(m1, h4.rename("h4_close").to_frame(), left_index=True, right_index=True)
