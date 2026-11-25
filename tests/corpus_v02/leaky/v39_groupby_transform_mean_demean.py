# expect: LA09:medium
import pandas as pd
panel = pd.read_csv("panel.csv", parse_dates=["date"]).sort_values(["ticker", "date"])
panel["ret"] = panel.groupby("ticker")["close"].pct_change()
panel["ret_vs_avg"] = panel["ret"] - panel.groupby("ticker")["ret"].transform("mean")
panel["signal"] = (panel["ret_vs_avg"] < -0.02).astype(int)
