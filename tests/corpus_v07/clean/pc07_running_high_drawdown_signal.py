# expect: none
import pandas as pd

px = pd.read_csv("qqq.csv", index_col=0, parse_dates=True)["close"]
dd = px / px.cummax() - 1
buy = (dd < -0.25).astype(int)
strat = buy.shift(1) * px.pct_change()
print((1 + strat.fillna(0)).prod())
