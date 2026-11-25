import pandas as pd
panel = pd.read_csv("panel.csv", parse_dates=["date"])
panel["ret"] = panel.groupby("ticker")["close"].pct_change()
panel["z"] = panel.groupby("ticker")["ret"].transform(lambda s: (s - s.mean()) / s.std())
panel["signal"] = (panel["z"] < -2).astype(int)
panel["signal_lag"] = panel.groupby("ticker")["signal"].shift(1)
panel["strat"] = panel["signal_lag"] * panel["ret"]
