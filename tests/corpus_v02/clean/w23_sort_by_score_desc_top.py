# expect: none
import pandas as pd
panel = pd.read_csv("panel.csv", parse_dates=["date"]).sort_values(["ticker", "date"])
panel["mom"] = panel.groupby("ticker")["close"].pct_change(20)
last = panel[panel["date"] == panel["date"].max()]
top = last.sort_values("mom", ascending=False).head(20)
print(top["ticker"].tolist())
