# expect: none
import pandas as pd
panel = pd.read_csv("panel.csv", parse_dates=["date"])
last_seen = panel.groupby("ticker")["date"].max()
n_delisted = int((last_seen < panel["date"].max()).sum())
print(f"{n_delisted} names stop trading before the end and are kept with their final returns")
latest = panel[panel["date"] == panel["date"].max()]
print(latest["ticker"].nunique())
