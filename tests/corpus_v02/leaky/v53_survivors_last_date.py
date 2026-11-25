# expect: SB01:medium
import pandas as pd
panel = pd.read_csv("panel.csv", parse_dates=["date"]).sort_values(["ticker", "date"])
last_seen = panel.groupby("ticker")["date"].max()
alive = last_seen[last_seen == panel["date"].max()].index
panel = panel[panel["ticker"].isin(alive)]
panel["mom"] = panel.groupby("ticker")["close"].pct_change(126)
panel["rank"] = panel.groupby("date")["mom"].rank(pct=True)
