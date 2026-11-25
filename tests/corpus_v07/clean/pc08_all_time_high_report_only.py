# expect: none
import pandas as pd

px = pd.read_csv("qqq.csv", index_col=0, parse_dates=True)["close"]
ath = px.max()
print("currently", round((px.iloc[-1] / ath - 1) * 100, 1), "% below the all-time high")
