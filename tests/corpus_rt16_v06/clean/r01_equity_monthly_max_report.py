# expect: none
import pandas as pd
e = pd.read_csv("equity_curve.csv", index_col=0, parse_dates=True)
e["equity_month_peak"] = e.groupby(e.index.to_period("M"))["equity"].transform("max")
e["month_dd"] = e["equity"] / e["equity_month_peak"] - 1
print(e["month_dd"].groupby(e.index.to_period("M")).min())
