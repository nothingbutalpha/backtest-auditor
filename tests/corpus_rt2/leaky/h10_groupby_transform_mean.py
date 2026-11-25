import pandas as pd
panel = pd.read_csv("panel.csv", parse_dates=["date"])
panel["ret"] = panel.groupby("ticker")["close"].pct_change()
panel["avg_ret"] = panel.groupby("ticker")["ret"].transform("mean")
panel["long"] = (panel["avg_ret"] > 0).astype(int)
panel["strat"] = panel.groupby("ticker")["long"].shift(1) * panel["ret"]
