# expect: none
import pandas as pd
panel = pd.read_csv("panel.csv", parse_dates=["date"]).sort_values(["ticker", "date"])
panel["mom"] = panel.groupby("ticker")["close"].pct_change(126)
latest = panel[panel["date"] == panel["date"].max()].copy()
latest["mom_rank"] = latest["mom"].rank(ascending=False)
latest["buy"] = latest["mom_rank"] <= 10
print(latest.loc[latest["buy"], "ticker"].tolist())
